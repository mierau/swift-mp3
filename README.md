# SwiftMP3

A pure Swift MP3 encoder with no external dependencies beyond Apple's Accelerate framework. Encode PCM audio to MPEG-1 Layer III (MP3) entirely in Swift.

## Features

- MPEG-1 Layer III encoding at 32kHz, 44.1kHz, and 48kHz sample rates
- CBR and VBR encoding modes
- Mono, stereo, and joint stereo
- Streaming encoding — feed samples incrementally
- Async/await support with `AsyncSequence` input
- File encoding with automatic Xing header
- Xing/Info header generation for accurate seeking
- Psychoacoustic model with masking thresholds
- Transient detection with short/long/mixed block switching
- ISO 11172-3 compliant Huffman coding and polyphase filterbank
- Uses Accelerate (vDSP/vForce) for vectorized DSP operations
- Fully `Sendable` — safe to use from any concurrency context

## Requirements

- Swift 5.10+
- macOS 13+, iOS 16+, tvOS 16+, watchOS 9+, visionOS 1+

## Installation

Add SwiftMP3 to your `Package.swift`:

```swift
dependencies: [
  .package(url: "https://github.com/mierau/swift-mp3.git", from: "1.0.0")
]
```

Then add `"SwiftMP3"` as a dependency of your target:

```swift
.target(
  name: "MyApp",
  dependencies: ["SwiftMP3"]
)
```

## Usage

### Basic Encoding

Use `newSession()` to create a mutable encoding session for synchronous, incremental encoding:

```swift
import SwiftMP3

let encoder = MP3Encoder(options: MP3EncoderOptions(
  sampleRate: 44_100,
  bitrateKbps: 128,
  mode: .stereo
))
var session = encoder.newSession()

// Feed interleaved PCM float samples (L, R, L, R, ...)
var mp3Data = session.appendSamples(pcmSamples)

// Flush any remaining buffered samples
mp3Data.append(session.flush())

// Generate a Xing header for accurate seeking (prepend to file)
let xingHeader = session.makeXingHeader()

// Write the complete MP3 file
var file = Data()
file.append(xingHeader)
file.append(mp3Data)
```

### Streaming Encode

Use `encode(_:)` with any `AsyncSequence` of `[Float]` to get an `AsyncThrowingStream` of MP3 frame data:

```swift
let encoder = MP3Encoder(options: options)

for try await frame in encoder.encode(sampleSource) {
  // Each `frame` is a Data containing one or more MP3 frames
  send(frame)
}
```

### File Encode

Use `encode(_:to:)` to write directly to a file, including the Xing header for seeking:

```swift
let encoder = MP3Encoder(options: options)
try await encoder.encode(sampleSource, to: outputURL)
```

### Live Microphone Recording

Feed audio from `AVAudioEngine` into an `AsyncStream` and encode to a file:

```swift
let engine = AVAudioEngine()
let input = engine.inputNode
let format = input.outputFormat(forBus: 0)

let source = AsyncStream<[Float]> { continuation in
  input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
    let samples = Array(UnsafeBufferPointer(
      start: buffer.floatChannelData?[0],
      count: Int(buffer.frameLength)
    ))
    continuation.yield(samples)
  }
  continuation.onTermination = { _ in
    input.removeTap(onBus: 0)
    engine.stop()
  }
}

try engine.start()
let encoder = MP3Encoder(options: MP3EncoderOptions(mode: .mono))
try await encoder.encode(source, to: recordingURL)
```

### Background Encoding

`MP3Encoder` is `Sendable`, so you can create it on the main actor and hand it off to a background task:

```swift
let encoder = MP3Encoder(options: options)

Task.detached {
  try await encoder.encode(sampleSource, to: outputURL)
}
```

### Network Streaming

Encode and send frames over a network connection:

```swift
let encoder = MP3Encoder(options: options)

for try await frame in encoder.encode(sampleSource) {
  try await connection.send(frame)
}
```

## API Reference

### `MP3EncoderOptions`

Configuration for the encoder.

| Property | Type | Default | Description |
|---|---|---|---|
| `sampleRate` | `Int` | `44_100` | Input sample rate (32000, 44100, or 48000) |
| `bitrateKbps` | `Int` | `128` | Target bitrate in kbps (32-320) |
| `vbr` | `Bool` | `false` | Enable variable bitrate encoding |
| `mode` | `Mode` | `.stereo` | Channel mode: `.mono`, `.stereo`, or `.jointStereo` |
| `quality` | `Int` | `5` | Encoding quality 0 (best) to 9 (fastest) |
| `crcProtected` | `Bool` | `false` | Add CRC error detection to frames |
| `original` | `Bool` | `true` | Set the original bit in frame headers |
| `copyright` | `Bool` | `false` | Set the copyright bit in frame headers |

### `MP3Encoder`

Stateless, `Sendable` encoder that holds configuration. Safe to share across tasks.

| Method | Description |
|---|---|
| `init(options:)` | Create an encoder with the given options |
| `newSession() -> EncoderSession` | Create a mutable encoding session for synchronous use |
| `encode(_:) -> AsyncThrowingStream<Data, Error>` | Stream MP3 frames from an `AsyncSequence` of sample buffers |
| `encode(_:to:) async throws` | Encode an `AsyncSequence` directly to a file with Xing header |

### `EncoderSession`

Mutable encoding session created via `MP3Encoder.newSession()`. Not `Sendable` — use from a single context.

| Method / Property | Description |
|---|---|
| `appendSamples(_: [Float]) -> Data` | Feed interleaved PCM samples, returns any complete MP3 frames |
| `flush() -> Data` | Pads and encodes any remaining buffered samples |
| `makeXingHeader() -> Data` | Generates a Xing/Info header frame (call after encoding is complete) |
| `encodedFrameCount: UInt32` | Number of MP3 frames encoded so far |
| `encodedByteCount: UInt32` | Total bytes of encoded audio data so far |

### Sample Format

Input samples are interleaved `Float` values in the range `-1.0` to `1.0`:

- **Mono**: `[s0, s1, s2, ...]`
- **Stereo / Joint Stereo**: `[L0, R0, L1, R1, L2, R2, ...]`

## License

MIT License. See [LICENSE](LICENSE) for details.
