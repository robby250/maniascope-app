// lazer_scores.js <copy of client.realm> [maps]  →  one JSON line per mania score (run by lazer_scores.py);
// with "maps", also one {"map": …} line per installed mania difficulty (recommendation availability)
const Realm = require("realm");
const r = new Realm({ path: process.argv[2], readOnly: true });
const mania = r.objects("Ruleset").filtered("ShortName == 'mania'")[0];
const scores = mania ? r.objects("Score").filtered("Ruleset == $0 && DeletePending == false", mania) : [];
const lines = [];
const out = (o) => lines.push(JSON.stringify(o));
for (const s of scores) {
  if (process.argv[3] === "recent" && s.Date < new Date(process.argv[4])) continue;
  const b = s.BeatmapInfo;
  const replay = [...s.Files].find(f => f.Filename === "replay.osr");
  out({
    id: String(s.ID), hash: s.BeatmapHash, acc: s.Accuracy, total: s.TotalScore, combo: s.MaxCombo,
    rank: s.Rank, date: s.Date, mods: s.Mods, stats: s.Statistics, user: s.User ? s.User.Username : null,
    user_id: s.User ? s.User.OnlineID : null, online_id: s.OnlineID, legacy_online_id: s.LegacyOnlineID,
    legacy: s.IsLegacyScore, client: s.ClientVersion, max_stats: s.MaximumStatistics,
    replay_sha256: replay && replay.File ? replay.File.Hash : null, pauses: [...s.Pauses],
    beatmap_id: b ? b.OnlineID : null, md5: b ? b.MD5Hash : null, status: b ? b.Status : null,
  });
}
if (process.argv[3] === "maps" && mania) {
  for (const b of r.objects("Beatmap").filtered("Ruleset == $0 && Hidden == false && BeatmapSet.DeletePending == false", mania)) {
    const m = b.Metadata || {};
    const audio = b.BeatmapSet && m.AudioFile && b.BeatmapSet.Files.find(
      f => (f.Filename || "").toLowerCase() === m.AudioFile.toLowerCase());
    out({ map: b.Hash, beatmap_id: b.OnlineID, set_id: b.BeatmapSet ? b.BeatmapSet.OnlineID : null, md5: b.MD5Hash,
          online_md5: b.OnlineMD5Hash, audio_sha: audio && audio.File ? audio.File.Hash : null,
          audio_required: Boolean(m.AudioFile), creator: m.Author ? m.Author.Username : null,
          status: b.Status, keys: b.Difficulty ? b.Difficulty.CircleSize : null, od: b.Difficulty ? b.Difficulty.OverallDifficulty : null,
          length: b.Length, title: m.Title, artist: m.Artist, version: b.DifficultyName, last_played: b.LastPlayed });
  }
}
lines.push("END");
// exit only once the pipe has taken everything; r.close() can hang on a read-only copy
process.stdout.write(lines.join("\n") + "\n", () => process.exit(0));
