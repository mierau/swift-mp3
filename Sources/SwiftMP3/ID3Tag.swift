// SwiftMP3
// ID3Tag.swift

import Foundation

/// ID3v2.3 metadata for MP3 files.
public struct ID3Tag: Sendable, Equatable {
  /// The track title.
  public var title: String?
  /// The artist name.
  public var artist: String?
  /// The album name.
  public var album: String?
  /// The track number.
  public var track: UInt16?
  /// The total number of tracks.
  public var trackTotal: UInt16?
  /// The release year.
  public var year: UInt16?
  /// The genre name.
  public var genre: String?
  /// A comment.
  public var comment: String?
  /// Album artwork image data (JPEG or PNG).
  public var albumArt: Data?
  /// MIME type for the album artwork.
  public var albumArtMIMEType: String

  /// Creates a new ID3 tag with the given metadata.
  public init(
    title: String? = nil,
    artist: String? = nil,
    album: String? = nil,
    track: UInt16? = nil,
    trackTotal: UInt16? = nil,
    year: UInt16? = nil,
    genre: String? = nil,
    comment: String? = nil,
    albumArt: Data? = nil,
    albumArtMIMEType: String = "image/jpeg"
  ) {
    self.title = title
    self.artist = artist
    self.album = album
    self.track = track
    self.trackTotal = trackTotal
    self.year = year
    self.genre = genre
    self.comment = comment
    self.albumArt = albumArt
    self.albumArtMIMEType = albumArtMIMEType
  }
}

/// Assembles ID3v2.3 binary tag data from an ``ID3Tag``.
enum ID3TagWriter {
  /// Builds a complete ID3v2.3 tag from the given metadata.
  /// Returns empty `Data` if no fields are set.
  static func build(tag: ID3Tag) -> Data {
    var frames = Data()

    if let title = tag.title { frames.append(textFrame(id: "TIT2", value: title)) }
    if let artist = tag.artist { frames.append(textFrame(id: "TPE1", value: artist)) }
    if let album = tag.album { frames.append(textFrame(id: "TALB", value: album)) }
    if let genre = tag.genre { frames.append(textFrame(id: "TCON", value: genre)) }

    if let year = tag.year {
      frames.append(textFrame(id: "TYER", value: String(year)))
    }

    if let track = tag.track {
      let value = tag.trackTotal.map { "\(track)/\($0)" } ?? "\(track)"
      frames.append(textFrame(id: "TRCK", value: value))
    }

    if let comment = tag.comment {
      frames.append(commentFrame(comment: comment))
    }

    if let art = tag.albumArt {
      frames.append(pictureFrame(art: art, mimeType: tag.albumArtMIMEType))
    }

    if frames.isEmpty { return Data() }

    // ID3v2.3 header: "ID3" + version 2.3 + no flags + synchsafe size
    var header = Data()
    header.append(contentsOf: [0x49, 0x44, 0x33])  // "ID3"
    header.append(contentsOf: [0x03, 0x00])  // Version 2.3
    header.append(0x00)  // Flags
    header.append(contentsOf: synchsafeSize(UInt32(frames.count)))

    return header + frames
  }

  /// ID3v2.3 text encodings: ISO-8859-1 when possible, otherwise UTF-16 with a BOM.
  /// (UTF-8 is only defined from ID3v2.4.)
  private enum TextEncoding: UInt8 {
    case latin1 = 0x00
    case utf16 = 0x01

    init(for strings: String...) {
      let latin1 = strings.allSatisfy { $0.unicodeScalars.allSatisfy { $0.value <= 0xFF } }
      self = latin1 ? .latin1 : .utf16
    }

    func encode(_ string: String) -> Data {
      switch self {
      case .latin1:
        return Data(string.unicodeScalars.map { UInt8($0.value) })
      case .utf16:
        var data = Data([0xFF, 0xFE])
        for unit in string.utf16 {
          data.append(UInt8(unit & 0xFF))
          data.append(UInt8(unit >> 8))
        }
        return data
      }
    }

    var terminator: Data {
      self == .latin1 ? Data([0x00]) : Data([0x00, 0x00])
    }
  }

  /// Builds a text frame (TIT2, TPE1, TALB, TCON, TYER, TRCK).
  private static func textFrame(id: String, value: String) -> Data {
    let encoding = TextEncoding(for: value)
    var content = Data([encoding.rawValue])
    content.append(encoding.encode(value))
    return frameHeader(id: id, size: UInt32(content.count)) + content
  }

  /// Builds a COMM (comment) frame.
  private static func commentFrame(comment: String) -> Data {
    let encoding = TextEncoding(for: comment)
    var content = Data([encoding.rawValue])
    content.append(contentsOf: [0x65, 0x6E, 0x67])  // "eng"
    content.append(encoding.encode(""))  // Empty description
    content.append(encoding.terminator)
    content.append(encoding.encode(comment))
    return frameHeader(id: "COMM", size: UInt32(content.count)) + content
  }

  /// Builds an APIC (attached picture) frame.
  private static func pictureFrame(art: Data, mimeType: String) -> Data {
    var content = Data([TextEncoding.latin1.rawValue])
    content.append(Data(mimeType.utf8))
    content.append(0x00)  // MIME terminator
    content.append(0x03)  // Picture type: front cover
    content.append(0x00)  // Empty description
    content.append(art)
    return frameHeader(id: "APIC", size: UInt32(content.count)) + content
  }

  /// Encodes an integer as a 4-byte synchsafe integer (7 bits per byte).
  private static func synchsafeSize(_ size: UInt32) -> [UInt8] {
    [
      UInt8((size >> 21) & 0x7F),
      UInt8((size >> 14) & 0x7F),
      UInt8((size >> 7) & 0x7F),
      UInt8(size & 0x7F),
    ]
  }

  /// Builds a 10-byte frame header: 4-char ID + 4-byte big-endian size + 2 flag bytes.
  private static func frameHeader(id: String, size: UInt32) -> Data {
    var header = Data(id.utf8)
    header.append(UInt8((size >> 24) & 0xFF))
    header.append(UInt8((size >> 16) & 0xFF))
    header.append(UInt8((size >> 8) & 0xFF))
    header.append(UInt8(size & 0xFF))
    header.append(contentsOf: [0x00, 0x00])  // No flags
    return header
  }
}
