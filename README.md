# SwiftMP3

A pure Swift MP3 encoder with no external dependencies beyond Apple's Accelerate framework. Encode PCM audio to MPEG-1 Layer III (MP3) entirely in Swift.

## Features

- MPEG-1 Layer III encoding at 32kHz, 44.1kHz, and 48kHz sample rates
- CBR and VBR encoding modes
- Mono, stereo, and joint stereo
- Streaming encoding — feed samples incrementally
- Xing/Info header generation for accurate seeking
- Psychoacoustic model with masking thresholds
- Transient detection with short/long/mixed block switching
- ISO 11172-3 compliant Huffman coding and polyphase filterbank
- Uses Accelerate (vDSP/vForce) for vectorized DSP operations

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

```swift
import SwiftMP3

// Configure the encoder
let options = MP3EncoderOptions(
  sampleRate: 44_100,
  bitrateKbps: 128,
  mode: .stereo
)
let encoder = MP3Encoder(options: options)

// Feed interleaved PCM float samples (L, R, L, R, ...)
let mp3Data = encoder.appendSamples(pcmSamples)

// Flush any remaining buffered samples
let finalData = encoder.flush()

// Generate a Xing header for accurate seeking (prepend to file)
let xingHeader = encoder.makeXingHeader()

// Write the complete MP3 file
var file = Data()
file.append(xingHeader)
file.append(mp3Data)
file.append(finalData)
```

### Streaming Encoding

`appendSamples(_:)` buffers input and returns encoded MP3 data as soon as full frames (1152 samples per channel) are available. Call it repeatedly with chunks of any size:

```swift
let encoder = MP3Encoder(options: options)
var mp3File = Data()

for chunk in audioChunks {
  mp3File.append(encoder.appendSamples(chunk))
}
mp3File.append(encoder.flush())

// Prepend Xing header for seekable output
let complete = encoder.makeXingHeader() + mp3File
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

The encoder instance. Not thread-safe — use from a single task or queue.

| Method / Property | Description |
|---|---|
| `init(options:)` | Create an encoder with the given options |
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
