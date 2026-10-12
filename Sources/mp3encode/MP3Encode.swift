// SwiftMP3
// mp3encode/MP3Encode.swift
//
// Encodes an audio file to MP3 from the command line:
//
//   swift run -c release mp3encode input.wav [output.mp3] [options]
//
// Apple platforms read any format AVAudioFile does; elsewhere, input must be WAV.

import Foundation
import SwiftMP3

let usage = """
  Usage: mp3encode <input> [<output.mp3>] [options]

  Options:
    -b, --bitrate <kbps>   Bitrate, or the maximum average with --vbr (default 128)
    --vbr                  Vary the bitrate frame by frame
    -q, --quality <0-9>    0 is best, 9 is fastest (default 5)
    -m, --mode <mode>      mono, stereo, or joint (default joint)
    -r, --rate <hz>        MP3 sample rate (default: picked from the input)
    --title <text>         ID3 title
    --artist <text>        ID3 artist
    --album <text>         ID3 album
    --album-artist <text>  ID3 album artist
    --year <year>          ID3 year
    --track <n[/total]>    ID3 track number
    --genre <text>         ID3 genre
    --art <image>          ID3 front cover, JPEG or PNG
    -h, --help             Show this help
  """

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("mp3encode: \(message)\n".utf8))
  exit(1)
}

@main
struct MP3Encode {
  static func main() async {
    var positional: [String] = []
    var options = MP3EncoderOptions()
    var tag = ID3Tag()
    var arguments = CommandLine.arguments.dropFirst()

    func nextValue(for flag: String) -> String {
      guard let value = arguments.popFirst() else { fail("\(flag) needs a value") }
      return value
    }

    func nextInt(for flag: String) -> Int {
      let value = nextValue(for: flag)
      guard let number = Int(value) else { fail("\(flag) expects a number, got \(value)") }
      return number
    }

    while let argument = arguments.popFirst() {
      switch argument {
      case "-h", "--help":
        print(usage)
        exit(0)
      case "-b", "--bitrate": options.bitrateKbps = nextInt(for: argument)
      case "--vbr": options.vbr = true
      case "-q", "--quality": options.quality = nextInt(for: argument)
      case "-r", "--rate": options.outputSampleRate = nextInt(for: argument)
      case "-m", "--mode":
        switch nextValue(for: argument) {
        case "mono": options.mode = .mono
        case "stereo": options.mode = .stereo
        case "joint": options.mode = .jointStereo
        case let mode: fail("unknown mode \(mode); expected mono, stereo, or joint")
        }
      case "--title": tag.title = nextValue(for: argument)
      case "--artist": tag.artist = nextValue(for: argument)
      case "--album": tag.album = nextValue(for: argument)
      case "--album-artist": tag.albumArtist = nextValue(for: argument)
      case "--genre": tag.genre = nextValue(for: argument)
      case "--year": tag.year = UInt16(nextInt(for: argument))
      case "--track":
        let parts = nextValue(for: argument).split(separator: "/").map { UInt16($0) }
        guard let track = parts.first ?? nil, parts.count <= 2 else { fail("--track expects n or n/total") }
        tag.track = track
        tag.trackTotal = parts.count == 2 ? parts[1] : nil
      case "--art":
        let path = nextValue(for: argument)
        guard let art = FileManager.default.contents(atPath: path) else { fail("can't read \(path)") }
        tag.albumArt = art
        tag.albumArtMIMEType = art.starts(with: [0x89, 0x50, 0x4E, 0x47]) ? "image/png" : "image/jpeg"
      case let flag where flag.hasPrefix("-") && flag.count > 1:
        fail("unknown option \(flag)\n\n\(usage)")
      default:
        positional.append(argument)
      }
    }

    guard (1...2).contains(positional.count) else { fail("expected an input file\n\n\(usage)") }
    let input = URL(fileURLWithPath: positional[0])
    let output = positional.count == 2
      ? URL(fileURLWithPath: positional[1])
      : input.deletingPathExtension().appendingPathExtension("mp3")
    if tag != ID3Tag() { options.id3Tag = tag }

    do {
      let encoder = try MP3Encoder(options: options)
      let start = Date()
      try await encoder.encode(contentsOf: input, to: output) { fraction in
        FileHandle.standardError.write(Data("\rEncoding \(input.lastPathComponent): \(Int(fraction * 100))%".utf8))
      }
      let seconds = Date().timeIntervalSince(start)
      FileHandle.standardError.write(Data("\rWrote \(output.path) in \(String(format: "%.1f", seconds)) s\u{1B}[K\n".utf8))
    } catch {
      FileHandle.standardError.write(Data("\n".utf8))
      fail((error as? MP3EncoderError)?.description ?? error.localizedDescription)
    }
  }
}
