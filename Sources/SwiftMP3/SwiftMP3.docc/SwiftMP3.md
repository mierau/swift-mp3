# ``SwiftMP3``

Encode PCM audio to MP3 in pure Swift.

## Overview

SwiftMP3 is an MPEG Layer III encoder written entirely in Swift with no dependencies. It supports MPEG-1, MPEG-2, and MPEG-2.5 sample rates, CBR and VBR encoding, mono, stereo, and joint stereo, ID3v2.3 metadata, and Xing/Info and LAME tags for accurate seeking and gapless playback.

Create an ``MP3Encoder`` with a set of ``MP3EncoderOptions``, then encode synchronously with an ``EncoderSession``, or asynchronously from any `AsyncSequence` of sample buffers.

```swift
import SwiftMP3

let encoder = try MP3Encoder(options: MP3EncoderOptions(
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

Sessions also accept 16-bit integer samples, one array per channel, and `AVAudioPCMBuffer`.

Supported sample rates are 32, 44.1, and 48 kHz (MPEG-1), 16, 22.05, and 24 kHz (MPEG-2), and 8, 11.025, and 12 kHz (MPEG-2.5).

## Topics

### Encoding

- ``MP3Encoder``
- ``MP3EncoderOptions``
- ``EncoderSession``
- ``MP3EncoderError``

### Metadata

- ``ID3Tag``
