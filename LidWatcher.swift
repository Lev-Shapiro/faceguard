import Cocoa
import AVFoundation
import Vision

let home = FileManager.default.homeDirectoryForCurrentUser.path
let baseDir = "\(home)/.faceguard"
let meDir = "\(baseDir)/me"
let intrudersDir = "\(baseDir)/intruders"
let logPath = "\(baseDir)/watcher.log"
let capturePath = "\(baseDir)/latest_capture.jpg"

// Vision FeaturePrint distance threshold (Lower = closer match. Typical same-person cropped face is 0.20 - 0.52)
let DISTANCE_THRESHOLD: Float = 0.52

func log(_ message: String) {
    let line = "[\(Date())] \(message)\n"
    print(line, terminator: "")
    if let data = line.data(using: .utf8) {
        if FileManager.default.fileExists(atPath: logPath),
           let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logPath)) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: URL(fileURLWithPath: logPath))
        }
    }
}

class NativeFaceGuard: NSObject, AVCapturePhotoCaptureDelegate {
    private var session: AVCaptureSession?
    private var output: AVCapturePhotoOutput?
    private var isCapturing = false
    private var lastTrigger = Date.distantPast
    private var ownerPrints: [VNFeaturePrintObservation] = []

    // Extracts a tightly cropped face CGImage and generates a Neural Engine FeaturePrint
    func extractFacePrint(from cgImage: CGImage) -> VNFeaturePrintObservation? {
        let faceReq = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([faceReq])

        guard let face = (faceReq.results as? [VNFaceObservation])?.first else {
            return nil
        }

        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let box = face.boundingBox

        // Convert Vision bottom-left normalized coordinates to CGImage top-left pixel coordinates
        let rect = CGRect(
            x: box.origin.x * w,
            y: (1.0 - box.origin.y - box.height) * h,
            width: box.width * w,
            height: box.height * h
        ).insetBy(dx: -box.width * w * 0.15, dy: -box.height * h * 0.15)
         .intersection(CGRect(x: 0, y: 0, width: w, height: h))

        guard let faceCrop = cgImage.cropping(to: rect) else { return nil }

        let printReq = VNGenerateImageFeaturePrintRequest()
        let cropHandler = VNImageRequestHandler(cgImage: faceCrop, options: [:])
        try? cropHandler.perform([printReq])

        return (printReq.results as? [VNFeaturePrintObservation])?.first
    }

    func enrollOwnerPhotos() {
        ownerPrints.removeAll()
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: meDir) else {
            log("ERROR: Could not read \(meDir)")
            return
        }

        for file in files where !file.hasPrefix(".") {
            let path = "\(meDir)/\(file)"
            guard let nsImage = NSImage(contentsOfFile: path),
                  let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                continue
            }
            if let fp = extractFacePrint(from: cgImage) {
                ownerPrints.append(fp)
            }
        }
        log("Enrolled \(ownerPrints.count) reference photos from ~/.faceguard/me/ in memory.")
    }

    func start() {
        try? FileManager.default.createDirectory(atPath: intrudersDir, withIntermediateDirectories: true)
        enrollOwnerPhotos()

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            log("Camera permission verified.")
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                log("Camera permission prompt result: \(granted)")
            }
        default:
            log("ERROR: Camera permission denied in System Settings.")
        }

        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.screensDidWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.trigger(reason: "screensDidWake") }

        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.trigger(reason: "systemDidWake") }

        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name("com.apple.screenIsUnlocked"), object: nil, queue: .main
        ) { [weak self] _ in self?.trigger(reason: "screenIsUnlocked") }
    }

    func trigger(reason: String) {
        guard Date().timeIntervalSince(lastTrigger) > 6.0, !isCapturing else { return }
        lastTrigger = Date()
        isCapturing = true
        log("Triggered by [\(reason)]. Waiting 1.0s for camera...")

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.capturePhoto()
        }
    }

    func capturePhoto() {
        guard let device = AVCaptureDevice.default(for: .video) else {
            log("ERROR: No camera found.")
            isCapturing = false
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            let session = AVCaptureSession()
            session.sessionPreset = .photo
            let output = AVCapturePhotoOutput()

            if session.canAddInput(input) && session.canAddOutput(output) {
                session.addInput(input)
                session.addOutput(output)
                self.session = session
                self.output = output
                session.startRunning()

                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    output.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
                }
            }
        } catch {
            log("Camera error: \(error)")
            isCapturing = false
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        session?.stopRunning()
        session = nil
        defer { isCapturing = false }

        guard let data = photo.fileDataRepresentation(),
              let nsImage = NSImage(data: data),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            log("ERROR: Failed to decode captured frame.")
            return
        }

        try? data.write(to: URL(fileURLWithPath: capturePath))
        verifyFace(cgImage: cgImage, rawData: data)
    }

    func verifyFace(cgImage: CGImage, rawData: Data) {
        var minDistance: Float = 999.0
        var isOwner = false

        if let livePrint = extractFacePrint(from: cgImage) {
            for refPrint in ownerPrints {
                var dist: Float = 999.0
                try? livePrint.computeDistance(&dist, to: refPrint)
                if dist < minDistance {
                    minDistance = dist
                }
            }
            if minDistance <= DISTANCE_THRESHOLD {
                isOwner = true
            }
        }

        if isOwner {
            log("Result: OWNER VERIFIED (minDistance: \(String(format: "%.3f", minDistance)) <= \(DISTANCE_THRESHOLD))")
        } else {
            log("Result: INTRUDER ALERT! (minDistance: \(String(format: "%.3f", minDistance)))")
            dispatchIntruderAlert(rawData: rawData, distance: minDistance)
        }
    }

    func dispatchIntruderAlert(rawData: Data, distance: Float) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let ts = formatter.string(from: Date())
        let distStr = distance > 100 ? "no_face" : String(format: "%.2f", distance)
        let filename = "intruder_\(ts)_dist_\(distStr).jpg"

        let localFile = "\(intrudersDir)/\(filename)"
        try? rawData.write(to: URL(fileURLWithPath: localFile))

        let fm = FileManager.default
        // 1. Sync to iCloud Drive
        let icloudDir = "\(home)/Library/Mobile Documents/com~apple~CloudDocs/SecurityAlerts"
        try? fm.createDirectory(atPath: icloudDir, withIntermediateDirectories: true)
        try? fm.copyItem(atPath: localFile, toPath: "\(icloudDir)/\(filename)")

        // 2. Sync to Google Drive (if mounted)
        let cloudStorage = "\(home)/Library/CloudStorage"
        if let mounts = try? fm.contentsOfDirectory(atPath: cloudStorage) {
            for mount in mounts where mount.hasPrefix("GoogleDrive-") {
                let gdriveDir = "\(cloudStorage)/\(mount)/My Drive/SecurityAlerts"
                try? fm.createDirectory(atPath: gdriveDir, withIntermediateDirectories: true)
                try? fm.copyItem(atPath: localFile, toPath: "\(gdriveDir)/\(filename)")
            }
        }
    }
}

let app = NSApplication.shared
let guardApp = NativeFaceGuard()
guardApp.start()
if CommandLine.arguments.contains("--test") {
    guardApp.trigger(reason: "manualStartupTest")
}
app.run()
