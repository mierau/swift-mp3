# ``SwiftMP3``

Encode PCM audio to MP3 in pure Swift.

## Overview

SwiftMP3 is an MPEG-1 Layer III encoder written entirely in Swift with no dependencies. It supports CBR and VBR encoding, mono, stereo, and joint stereo, ID3v2.3 metadata, and Xing/Info headers for accurate seeking.

Create an ``MP3Encoder`` with a set of ``MP3EncoderOptions``, then encode synchronously with an ``EncoderSession``, or asynchronously from any `AsyncSequence` of sample buffers.

```swift
import SwiftMP3

let encoder = MP3Encoder(options: MP3EncoderOptions(
  sampleRate: 44_100,
  bitrateKbps: 192,
  mode: .jointStereo
))

try await encoder.encode(samples, to: outputURL)
```

### Sample Format

Input samples are interleaved `Float` values in the range `-1.0` to `1.0`:

- **Mono**: `[s0, s1, s2, ...]`
- **Stereo / Joint Stereo**: `[L0, R0, L1, R1, ...]`

Supported sample rates are 32 kHz, 44.1 kHz, and 48 kHz.

## Topics

### Encoding

- ``MP3Encoder``
- ``MP3EncoderOptions``
- ``EncoderSession``

### Metadata

- ``ID3Tag``
