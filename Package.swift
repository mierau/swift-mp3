// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "swift-mp3",
  platforms: [
    .macOS(.v13),
    .iOS(.v16),
    .tvOS(.v16),
    .watchOS(.v9),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "SwiftMP3",
      targets: ["SwiftMP3"]
    ),
  ],
  targets: [
    .target(
      name: "SwiftMP3"
    ),
    .executableTarget(
      name: "SwiftMP3Benchmark",
      dependencies: ["SwiftMP3"]
    ),
    .testTarget(
      name: "SwiftMP3Tests",
      dependencies: ["SwiftMP3"]
    ),
  ]
)
