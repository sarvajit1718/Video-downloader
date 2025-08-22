"""Metadata extractor with simplified format selection + compact persistent cache.

Key behavior & upgrades:
- Keeps original selection and scoring logic (video pairing, MAIN_HEIGHT resolution selection).
- Persistent *compact* cache saved to `config/metadata_cache.json` when `cache_raw=True`.
  The cache stores a compacted version of the original yt-dlp info (only fields needed
  for selection/printing) to avoid 10k+ line raw dumps.
- `force_refresh` option to bypass the cache and fetch fresh metadata.
- Richer error dicts with `stage` and `url`.
- `pretty_print(..., style=...)` supports 'plain' and 'json' output.
- Config loader that gracefully falls back if `settings.toml` is absent.

The idea: keep the original algorithm identical but persist a much smaller
representation of the metadata so the cache remains useful and small.
"""
from __future__ import annotations

import json
import os
import datetime
import time
from typing import Dict, List, Any, Optional, Tuple

try:
    # Python 3.11+
    import tomllib as _toml_loader
except Exception:
    try:
        import toml as _toml_loader  # type: ignore
    except Exception:
        _toml_loader = None  # type: ignore

import yt_dlp

MAIN_HEIGHTS = [240, 360, 480, 720, 1080, 1440, 2160]
DEFAULT_CACHE_FILE = os.path.join("config", "metadata_cache.json")
DEFAULT_SETTINGS_FILE = os.path.join("config", "settings.toml")


# ---------- helpers ----------
def _format_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "Live / Unknown"
    h = seconds // 3600
    m = (seconds // 60) % 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _human_readable_size(size_bytes: Optional[int]) -> str:
    if not size_bytes:
        return "Unknown"
    size = float(size_bytes)
    units = ("B", "KiB", "MiB", "GiB")
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _get_height_from_format(f: Dict[str, Any]) -> Optional[int]:
    h = f.get("height")
    if isinstance(h, int):
        return h
    res = f.get("resolution")
    if isinstance(res, str) and "x" in res:
        try:
            return int(res.split("x")[1])
        except Exception:
            return None
    return None


def _filesize_of(f: Optional[Dict[str, Any]]) -> Optional[int]:
    if not f:
        return None
    return f.get("filesize") or f.get("filesize_approx")


def _is_audio_only(f: Dict[str, Any]) -> bool:
    v = f.get("vcodec")
    return (not v) or str(v).lower() in ("none", "n/a")


def _is_video_only(f: Dict[str, Any]) -> bool:
    v = f.get("vcodec")
    a = f.get("acodec")
    if not v or str(v).lower() in ("none", "n/a"):
        return False
    if a and str(a).lower() not in ("none", "n/a"):
        return False
    return True


def _metric_bitrate_or_filesize(f: Dict[str, Any]) -> float:
    val = f.get("abr") or f.get("tbr")
    if val:
        try:
            return float(val)
        except Exception:
            pass
    fs = _filesize_of(f)
    if fs:
        return float(fs) / (1024 * 1024)
    return 0.0


def _score_video_candidate(f: Dict[str, Any], preferred_exts: List[str]) -> float:
    score = 0.0
    score += _metric_bitrate_or_filesize(f)

    vcodec = (f.get("vcodec") or "").lower()
    if "av1" in vcodec or "av01" in vcodec:
        score += 200.0
    elif "vp9" in vcodec:
        score += 120.0
    elif "h264" in vcodec or "avc" in vcodec:
        score += 40.0

    fps = f.get("fps") or 0
    try:
        fps = int(fps)
    except Exception:
        fps = 0
    if fps >= 60:
        score += 80.0
    elif fps >= 48:
        score += 40.0

    height = _get_height_from_format(f) or 0
    score += height / 100.0

    for idx, ext in enumerate(preferred_exts):
        if f.get("ext") == ext:
            score += (len(preferred_exts) - idx) * 10.0
            break

    return score


# ---------- size estimation helper ----------
def _estimate_size_bytes(fmt: Optional[Dict[str, Any]], duration: Optional[int] = None) -> Optional[int]:
    if not fmt:
        return None
    if fmt.get("filesize"):
        try:
            return int(fmt["filesize"])
        except Exception:
            pass
    if fmt.get("filesize_approx"):
        try:
            return int(fmt["filesize_approx"])
        except Exception:
            pass
    tbr = fmt.get("tbr") or fmt.get("abr")
    if tbr and duration:
        try:
            return int(float(tbr) * 1000.0 * float(duration) / 8.0)
        except Exception:
            return None
    return None


# ---------- config & cache helpers ----------

def _ensure_config_dir_exists():
    cfg_dir = os.path.dirname(DEFAULT_CACHE_FILE)
    if not cfg_dir:
        return
    os.makedirs(cfg_dir, exist_ok=True)


def _load_json_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_json_cache(path: str, data: Dict[str, Any]) -> None:
    _ensure_config_dir_exists()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _load_settings(path: str = DEFAULT_SETTINGS_FILE) -> Dict[str, Any]:
    if not _toml_loader:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as fh:
            if hasattr(_toml_loader, "load"):
                return _toml_loader.load(fh)  # type: ignore
            else:
                return _toml_loader.loads(fh.read().decode("utf-8"))  # type: ignore
    except Exception:
        return {}


# compacting raw info before persisting made small and contains fields needed by the original logic
def _compact_info_for_cache(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a refined cache-friendly dict with the structure:
    {
      "id": "abc123xyz",
      "title": "Some Video Title",
      "uploader": "Channel Name",
      "duration": 532,
      "upload_date": "2025-08-21",
      "view_count": 1234567,
      "like_count": 4321,
      "thumbnail": "https://...",
      "audio": {
        "m4a": {"format_id": "140", "ext": "m4a", "filesize": 3558290, "bitrate": 128},
        "opus": {"format_id": "251", "ext": "webm", "filesize": 2940192, "bitrate": 160}
      },
      "video": {
        "mp4": [
          {"resolution": "240p", "format_id": "134", "ext": "mp4", "filesize": 5328290, "height": 240, "width": 426, "fps": 30, "vcodec": "avc1.4d4015", "acodec": "none"},
          ...
        ],
        "webm": [
          {"resolution": "240p", "format_id": "243", "ext": "webm", "filesize": 4728290, "height": 240, "width": 426, "fps": 30, "vcodec": "vp9", "acodec": "none"},
          ...
        ]
      },
      "_last_updated": 1692641043
    }
    """
    comp: Dict[str, Any] = {}

    # --- top-level fields (minimal) ---
    for k in ("id", "title", "uploader", "view_count", "like_count", "thumbnail"):
        if k in info:
            comp[k] = info[k]

    # raw duration in seconds (keep number; UI can format)
    if "duration" in info:
        comp["duration"] = info.get("duration")

    # normalized upload_date -> YYYY-MM-DD
    raw_date = info.get("upload_date")
    if raw_date and isinstance(raw_date, str) and len(raw_date) >= 8:
        comp["upload_date"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    elif raw_date:
        comp["upload_date"] = raw_date

    # --- prepare formats list ---
    formats = info.get("formats") or []

    # audio-only candidates (that have at least filesize/abr/tbr)
    audio_candidates = [
        f for f in formats
        if _is_audio_only(f) and (_filesize_of(f) or f.get("abr") or f.get("tbr"))
    ]

    def _pick_best_audio_by_ext(ext: str) -> Optional[Dict[str, Any]]:
        matches = [a for a in audio_candidates if (a.get("ext") or "").lower() == ext]
        if not matches:
            return None
        return max(matches, key=lambda x: ((x.get("abr") or x.get("tbr") or 0), _filesize_of(x) or 0))

    best_m4a = _pick_best_audio_by_ext("m4a")
    best_opus = _pick_best_audio_by_ext("opus")

    comp["audio"] = {}
    if best_m4a:
        comp["audio"]["m4a"] = {
            "format_id": best_m4a.get("format_id"),
            "ext": best_m4a.get("ext"),
            "filesize": _filesize_of(best_m4a),
            "bitrate": best_m4a.get("abr") or best_m4a.get("tbr"),
            "url": best_m4a.get("url"),  # Store URL but note it will expire
        }
    if best_opus:
        comp["audio"]["opus"] = {
            "format_id": best_opus.get("format_id"),
            "ext": best_opus.get("ext"),
            "filesize": _filesize_of(best_opus),
            "bitrate": best_opus.get("abr") or best_opus.get("tbr"),
            "url": best_opus.get("url"),  # Store URL but note it will expire
        }

    # --- video: for each container (mp4, webm) store available resolutions ---
    comp["video"] = {"mp4": [], "webm": []}

    # Process all video formats
    for container in ("mp4", "webm"):
        # Get all video-only formats for this container
        video_formats = [
            f for f in formats
            if _is_video_only(f) and (f.get("ext") or "").lower() == container
            and (_filesize_of(f) or f.get("tbr") or f.get("fps"))
        ]
        
        # Group by height
        by_height = {}
        for f in video_formats:
            height = _get_height_from_format(f)
            if height is None:
                continue
                
            if height not in by_height:
                by_height[height] = []
            by_height[height].append(f)
        
        # For each height, pick the best format
        for height, formats_list in by_height.items():
            if height not in MAIN_HEIGHTS:
                continue
                
            # Score and select the best format for this height
            chosen = max(formats_list, key=lambda x: _score_video_candidate(x, ["mp4", "webm"]))
            
            # Get width from format or calculate from height and aspect ratio
            width = chosen.get("width")
            if not width and chosen.get("height") and chosen.get("aspect_ratio"):
                width = int(chosen["height"] * chosen["aspect_ratio"])
            elif not width:
                # Default aspect ratio 16:9
                width = int(height * 16 / 9)
            
            video_entry = {
                "format_id": chosen.get("format_id"),
                "ext": chosen.get("ext"),
                "filesize": _filesize_of(chosen),
                "resolution": f"{height}p",
                "height": height,
                "width": width,
                "fps": chosen.get("fps"),
                "vcodec": chosen.get("vcodec"),
                "acodec": "none",  # Video-only streams
                "url": chosen.get("url"),  # Store URL but note it will expire
            }
            comp["video"][container].append(video_entry)
        
        # Sort by resolution
        comp["video"][container].sort(key=lambda x: x["height"])

    # Add cache timestamp (Unix timestamp)
    comp["_last_updated"] = int(time.time())

    return comp


def _expand_compact_if_needed(compact_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert our refined cache format back to a format similar to original yt-dlp output."""
    # For our refined cache, we need to reconstruct a formats list
    # that the existing processing code can understand
    expanded = compact_info.copy()
    
    # Reconstruct formats list from our refined data
    formats = []
    
    # Add audio formats
    for audio_type, audio_info in compact_info.get("audio", {}).items():
        if audio_info:
            format_entry = {
                "format_id": audio_info["format_id"],
                "ext": audio_info["ext"],
                "filesize": audio_info["filesize"],
                "abr": audio_info.get("bitrate"),
                "tbr": audio_info.get("bitrate"),
                "vcodec": "none",
                "acodec": "mp4a.40.2" if audio_type == "m4a" else "opus",
                "url": audio_info.get("url"),
            }
            formats.append(format_entry)
    
    # Add video formats
    for container, video_list in compact_info.get("video", {}).items():
        for video_info in video_list:
            if video_info:
                format_entry = {
                    "format_id": video_info["format_id"],
                    "ext": video_info["ext"],
                    "filesize": video_info["filesize"],
                    "height": video_info["height"],
                    "width": video_info["width"],
                    "fps": video_info["fps"],
                    "vcodec": video_info["vcodec"],
                    "acodec": video_info["acodec"],
                    "resolution": f"{video_info['width']}x{video_info['height']}",
                    "url": video_info.get("url"),
                }
                formats.append(format_entry)
    
    expanded["formats"] = formats
    return expanded


# ---------- core extraction & selection ----------
class MetadataExtractor:
    def __init__(self, preferred_exts: Optional[List[str]] = None, cache_raw: bool = False, cache_file: Optional[str] = None):
        """
        preferred_exts: order in which container types should be favored (mp4 then webm typically).
        cache_raw: if True, caches compacted raw metadata for the same URL in-memory and on-disk (useful in interactive CLI).
        cache_file: path to persistent compact cache JSON file.
        """
        settings = _load_settings()
        cfg_pref = []
        try:
            cfg_pref = settings.get("download", {}).get("preferred_exts") or []
        except Exception:
            cfg_pref = []

        if preferred_exts is None:
            self.preferred_exts = cfg_pref or ["mp4", "webm"]
        else:
            self.preferred_exts = preferred_exts

        self._cache_raw = bool(cache_raw)
        self.cache_file = cache_file or DEFAULT_CACHE_FILE
        self._raw_cache: Dict[str, Dict[str, Any]] = {}

        if self._cache_raw:
            try:
                file_cache = _load_json_cache(self.cache_file)
                if isinstance(file_cache, dict):
                    self._raw_cache.update(file_cache)
            except Exception:
                self._raw_cache = {}

    def extract_raw(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Call yt-dlp to get raw metadata. Uses in-memory + on-disk compact cache when enabled.

        Raises RuntimeError on failure.
        """
        # Check if we have a valid cache entry that's not too old (less than 4 hours)
        if self._cache_raw and not force_refresh:
            if url in self._raw_cache:
                cached_data = self._raw_cache[url]
                cache_age = time.time() - cached_data.get("_last_updated", 0)
                
                # Use cache if it's less than 4 hours old (URLs expire after ~6 hours)
                if cache_age < 4 * 3600:
                    # Return the expanded cache data
                    return _expand_compact_if_needed(cached_data)
                else:
                    # Cache is stale, remove it
                    del self._raw_cache[url]

        # Fetch fresh data
        opts = {"quiet": True, "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            if self._cache_raw:
                try:
                    compact = _compact_info_for_cache(info)
                    # Store the compact form
                    self._raw_cache[url] = compact
                    file_cache = _load_json_cache(self.cache_file)
                    file_cache[url] = compact
                    _save_json_cache(self.cache_file, file_cache)
                except Exception as e:
                    print(f"Warning: Failed to save cache: {e}")
            
            return info
        except Exception as e:
            raise RuntimeError(f"yt-dlp failed to extract metadata: {e}") from e

    def extract_and_clean(self, url: str, mode: str = "video", force_refresh: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract metadata and return (cleaned_metadata, format_cache).
        format_cache layout:
          {
            "audio": [ { selector, ext, size_bytes, size_human, format_id }, ... ],
            "video": {
               "720p": { "mp4": {selector, size_bytes, size_human}, "webm": {...} },
               ...
            }
          }

        mode: 'video' or 'audio'.
        force_refresh: bypasses cache and fetches fresh metadata.
        """
        try:
            raw = self.extract_raw(url, force_refresh=force_refresh)
        except Exception as e:
            return ({"error": True, "stage": "metadata_extraction", "message": str(e), "url": url}, {})

        # If the cache returned compact info we need to handle it differently
        if raw.get("_type") == "playlist":
            entries_clean = []
            caches = {}
            for i, e in enumerate(raw.get("entries") or []):
                if not e:
                    continue
                try:
                    cleaned_e, cache_e = self._clean_single_with_cache(e, mode)
                    entries_clean.append(cleaned_e)
                    identifier = e.get("id") or f"entry_{i}"
                    caches[identifier] = cache_e
                except Exception:
                    entries_clean.append({"error": True, "title": e.get("title")})
            return ({"type": "playlist", "title": raw.get("title"), "entries": entries_clean}, caches)

        try:
            cleaned, cache = self._clean_single_with_cache(raw, mode)
            return (cleaned, cache)
        except Exception as e:
            return ({"error": True, "stage": "format_processing", "message": str(e), "url": url}, {})

    # ---------- core per-item processing ----------
    def _clean_single_with_cache(self, info: Dict[str, Any], mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cleaned: Dict[str, Any] = {
            "type": info.get("_type", "video"),
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader") or "Unknown",
            "duration": _format_duration(info.get("duration")),
            "upload_date": None,
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "thumbnail": info.get("thumbnail"),
            "formats": [],
        }
        raw_date = info.get("upload_date")
        if raw_date and isinstance(raw_date, str) and len(raw_date) >= 8:
            cleaned["upload_date"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

        duration_seconds = info.get("duration")
        formats = info.get("formats", []) or []

        audio_candidates = [f for f in formats if _is_audio_only(f) and (_filesize_of(f) or f.get("abr") or f.get("tbr"))]
        best_audio = self._pick_best_audio_global(audio_candidates)

        format_cache: Dict[str, Any] = {"audio": [], "video": {}}

        for h in sorted(MAIN_HEIGHTS, reverse=True):
            key = f"{h}p"
            format_cache["video"][key] = {"mp4": None, "webm": None}

            for container in ["mp4", "webm"]:
                candidates = [
                    f
                    for f in formats
                    if _is_video_only(f)
                    and _get_height_from_format(f) == h
                    and f.get("ext") == container
                    and (_filesize_of(f) or f.get("tbr") or f.get("fps"))
                ]

                if not candidates:
                    format_cache["video"][key][container] = None
                    continue

                chosen = max(candidates, key=lambda x: _score_video_candidate(x, self.preferred_exts))

                vid_bytes = _estimate_size_bytes(chosen, duration_seconds)
                aud_bytes = _estimate_size_bytes(best_audio, duration_seconds) if best_audio else None

                total_bytes = None
                if vid_bytes is not None and aud_bytes is not None:
                    total_bytes = int((vid_bytes or 0) + (aud_bytes or 0))
                elif vid_bytes is not None:
                    total_bytes = int(vid_bytes)
                elif aud_bytes is not None:
                    total_bytes = int(aud_bytes)

                size_human = _human_readable_size(total_bytes) if total_bytes is not None else "Unknown"

                if best_audio is None or chosen is None:
                    selector = None
                else:
                    selector = f"{chosen.get('format_id')}+{best_audio.get('format_id')}"

                format_cache["video"][key][container] = {
                    "selector": selector,
                    "size_bytes": total_bytes,
                    "size_human": size_human,
                    "video_format_id": chosen.get("format_id"),
                    "audio_format_id": best_audio.get("format_id") if best_audio else None,
                }

                cleaned["formats"].append(
                    {
                        "format_id": chosen.get("format_id"),
                        "ext": chosen.get("ext"),
                        "height": h,
                        "resolution": chosen.get("resolution") or f"{h}p",
                        "filesize": _filesize_of(chosen),
                        "size_human": _human_readable_size(_filesize_of(chosen)),
                        "format_selector": selector,
                        "paired_audio": best_audio.get("format_id") if best_audio else None,
                        "audio_ext": best_audio.get("ext") if best_audio else None,
                        "combined_size_bytes": total_bytes,
                        "combined_size_human": size_human,
                    }
                )

        # audio candidates for direct audio downloads
        cleaned["audio_candidates"] = []
        best_audio_list = self._top_audio_candidates(audio_candidates, top_n=6)
        for a in best_audio_list:
            audio_entry = {
                "format_id": a.get("format_id"),
                "ext": a.get("ext"),
                "bitrate": a.get("abr") or a.get("tbr"),
                "filesize": _filesize_of(a),
                "size_human": _human_readable_size(_filesize_of(a)),
                "format_selector": a.get("format_id"),
            }
            cleaned["audio_candidates"].append(audio_entry)
            # include mp3 and opus explicitly (if present in raw list they will be in audio_candidates)
            format_cache["audio"].append({
                "selector": a.get("format_id"),
                "ext": a.get("ext"),
                "size_bytes": _filesize_of(a),
                "size_human": _human_readable_size(_filesize_of(a)),
                "format_id": a.get("format_id"),
            })

        # remove empty resolutions
        empty_keys = [k for k, v in list(format_cache["video"].items()) if all(v.get(c) is None for c in ("mp4", "webm"))]
        for k in empty_keys:
            del format_cache["video"][k]

        return cleaned, format_cache

    # ---------- selection helpers ----------
    def _pick_best_audio_global(self, audio_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not audio_candidates:
            return None
        priority = ["m4a", "webm", "mp3"]
        for ext in priority:
            ext_matches = [a for a in audio_candidates if a.get("ext") == ext]
            if ext_matches:
                return max(ext_matches, key=lambda x: (x.get("abr") or x.get("tbr") or 0, _filesize_of(x) or 0))
        return max(audio_candidates, key=lambda x: (x.get("abr") or x.get("tbr") or 0, _filesize_of(x) or 0))

    def _top_audio_candidates(self, audio_candidates: List[Dict[str, Any]], top_n: int = 6) -> List[Dict[str, Any]]:
        if not audio_candidates:
            return []
        by_codec: Dict[str, Dict[str, Any]] = {}
        for a in audio_candidates:
            codec = a.get("acodec") or "unknown"
            rank = a.get("abr") or a.get("tbr") or _filesize_of(a) or 0
            if codec not in by_codec or rank > (by_codec[codec].get("abr") or by_codec[codec].get("tbr") or 0):
                by_codec[codec] = a
        candidates = sorted(list(by_codec.values()), key=lambda x: (x.get("abr") or x.get("tbr") or 0, _filesize_of(x) or 0), reverse=True)
        return candidates[:top_n]

    # ---------- pretty printing ----------
    def pretty_print(self, cleaned: Dict[str, Any], format_cache: Optional[Dict[str, Any]] = None, style: str = "plain") -> None:
        if style == "json":
            try:
                print(json.dumps(cleaned, indent=2, ensure_ascii=False))
            except Exception:
                print(str(cleaned))
            return

        if cleaned.get("type") == "playlist":
            print(f"Playlist: {cleaned.get('title')} (entries: {len(cleaned.get('entries', []))})")
            for i, e in enumerate(cleaned.get("entries", []), 1):
                print(f"--- Entry {i} ---")
                self.pretty_print(e, style=style)
            return

        if cleaned.get("error"):
            print("Error extracting metadata:")
            print(f"  stage  : {cleaned.get('stage')}")
            print(f"  url    : {cleaned.get('url')}")
            print(f"  message: {cleaned.get('message')}")
            return

        print("\n==== VIDEO INFO ====")
        print(f"Title      : {cleaned.get('title', 'Unknown')}")
        print(f"Uploader   : {cleaned.get('uploader', 'Unknown')}")
        print(f"Duration   : {cleaned.get('duration')}")
        print(f"Uploaded   : {cleaned.get('upload_date') or 'N/A'}")
        print(f"Views      : {cleaned.get('view_count') if cleaned.get('view_count') is not None else 'N/A'}")
        print(f"Likes      : {cleaned.get('like_count') if cleaned.get('like_count') is not None else 'N/A'}")
        print(f"Thumbnail  : {cleaned.get('thumbnail') or 'N/A'}")

        if format_cache:
            print("\n==== AVAILABLE FORMAT CACHE (resolution -> container -> selector (size)) ====")
            audio_list = format_cache.get("audio") or []
            if audio_list:
                print("-- Audio candidates --")
                for a in audio_list:
                    print(f"  - [{a.get('format_id')}] {a.get('ext')} -> {a.get('selector')} ({a.get('size_human')})")

            video_map = format_cache.get("video") or {}
            for res in sorted(video_map.keys(), key=lambda x: int(x.rstrip("p")), reverse=True):
                row = video_map[res]
                def label_for(entry):
                    if not entry:
                        return "-"
                    sel = entry.get("selector") or "-"
                    size = entry.get("size_human") or "Unknown"
                    return f"{sel} ({size})"
                mp4_label = label_for(row.get("mp4"))
                webm_label = label_for(row.get("webm"))
                print(f"{res:6} : mp4 -> {mp4_label:30} | webm -> {webm_label}")
        else:
            print("\n==== AVAILABLE FORMATS ====")
            fmts = cleaned.get("formats", [])
            if not fmts:
                print("No suitable formats found.")
                return
            for fmt in sorted(fmts, key=lambda x: x.get("height", 0), reverse=True):
                label = f"{fmt.get('height')}p" if fmt.get('height') else fmt.get('resolution')
                print(f"- [{fmt.get('format_id')}] {fmt.get('ext').upper()} {label} — {fmt.get('size_human')} (paired_audio: {fmt.get('paired_audio')})")


# ---------- simple CLI test ----------
if __name__ == "__main__":
    url = input("Enter a YouTube URL for metadata: ").strip()
    if not url:
        print("No URL provided — exiting.")
        raise SystemExit(1)

    me = MetadataExtractor(preferred_exts=["mp4", "webm"], cache_raw=True)
    cleaned_meta, cache = me.extract_and_clean(url, mode="video", force_refresh=False)
    me.pretty_print(cleaned_meta, cache)