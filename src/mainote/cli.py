"""Command-line interface for mainote."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from . import __version__
from .maidata.compiler import compile_maidata
from .maidata.model import Maidata
from .maidata.parser import parse_file
from .maidata.stats import chart_stats, maidata_stats
from .maidata.validate import validate_maidata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mainote",
        description="Audio -> chart IR -> maidata toolkit",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse maidata into IR JSON")
    p_parse.add_argument("maidata")
    p_parse.add_argument("--json", dest="json_out", help="write full IR JSON here")
    p_parse.set_defaults(func=_cmd_parse)

    p_compile = sub.add_parser("compile", help="compile IR JSON to maidata")
    p_compile.add_argument("ir")
    p_compile.add_argument("-o", "--out", required=True)
    p_compile.set_defaults(func=_cmd_compile)

    p_validate = sub.add_parser("validate", help="validate maidata or IR JSON")
    p_validate.add_argument("input")
    p_validate.add_argument("--json", action="store_true", help="print JSON report")
    p_validate.set_defaults(func=_cmd_validate)

    p_stats = sub.add_parser("stats", help="print chart statistics")
    p_stats.add_argument("input")
    p_stats.add_argument("--json", action="store_true", help="print JSON")
    p_stats.set_defaults(func=_cmd_stats)

    p_rt = sub.add_parser("roundtrip", help="IR -> maidata -> IR consistency check")
    p_rt.add_argument("maidata")
    p_rt.add_argument("--out", help="write the recompiled maidata here")
    p_rt.set_defaults(func=_cmd_roundtrip)

    p_eval = sub.add_parser("eval", help="compare a generated chart with a reference")
    p_eval.add_argument("generated")
    p_eval.add_argument("reference")
    p_eval.add_argument("--index", type=int, default=None, help="difficulty index (2..6)")
    p_eval.add_argument("--tolerance", type=float, default=0.06, help="time tolerance in seconds")
    p_eval.add_argument("--json", action="store_true", help="print JSON")
    p_eval.set_defaults(func=_cmd_eval)

    p_analyze = sub.add_parser("analyze", help="analyze audio timing")
    p_analyze.add_argument("audio")
    p_analyze.add_argument("--out", required=True, help="timing JSON output")
    p_analyze.add_argument("--sr", type=int, default=22050)
    p_analyze.add_argument("--bpm", type=float, default=None, help="override estimated BPM")
    p_analyze.add_argument("--first", type=float, default=None, help="chart start offset in seconds")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_gen = sub.add_parser("generate", help="generate a chart from audio")
    p_gen.add_argument("audio")
    p_gen.add_argument("-o", "--out", required=True, help="maidata output")
    p_gen.add_argument("--difficulty", default="master")
    p_gen.add_argument("--level", type=float, default=None)
    p_gen.add_argument(
        "--density",
        type=float,
        default=None,
        help="target notes per second (overrides the level/difficulty prior)",
    )
    p_gen.add_argument(
        "--density-scale",
        type=float,
        default=None,
        help="multiply the level/difficulty density target (e.g. 1.3)",
    )
    p_gen.add_argument(
        "--slide-prob",
        type=float,
        default=None,
        help="probability that a selected tick becomes a slide (0-1; default from difficulty)",
    )
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--sr", type=int, default=22050)
    p_gen.add_argument("--timing", help="use an existing timing JSON")
    p_gen.add_argument("--bpm", type=float, default=None, help="override timing BPM")
    p_gen.add_argument("--first", type=float, default=None, help="override chart start offset")
    p_gen.add_argument("--title", default=None)
    p_gen.add_argument("--artist", default=None)
    p_gen.add_argument("--ir", dest="ir_out", default=None, help="write IR JSON")
    p_gen.add_argument("--stats", dest="stats_out", default=None, help="write stats JSON")
    p_gen.set_defaults(func=_cmd_generate)

    p_dataset = sub.add_parser("dataset", help="dataset ingestion and reports")
    dsub = p_dataset.add_subparsers(dest="dataset_command", required=True)
    p_ingest = dsub.add_parser("ingest", help="scan raw song folders")
    p_ingest.add_argument("raw_dir")
    p_ingest.add_argument("--out", required=True, help="processed directory")
    p_ingest.add_argument("--align", action="store_true", help="run audio alignment scoring")
    p_ingest.add_argument("--sr", type=int, default=22050)
    p_ingest.set_defaults(func=_cmd_dataset_ingest)

    p_report = dsub.add_parser("report", help="aggregate processed metadata")
    p_report.add_argument("processed_dir")
    p_report.add_argument("--out", default=None, help="write report JSON")
    p_report.add_argument("--csv", default=None, help="write per-chart CSV")
    p_report.set_defaults(func=_cmd_dataset_report)

    p_split = dsub.add_parser("split", help="create leakage-safe splits by song")
    p_split.add_argument("processed_dir")
    p_split.add_argument("--out", required=True, help="write split JSON")
    p_split.add_argument("--train", type=float, default=0.8)
    p_split.add_argument("--val", type=float, default=0.1)
    p_split.add_argument("--test", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=0)
    p_split.set_defaults(func=_cmd_dataset_split)

    p_samples = dsub.add_parser(
        "samples", help="export one training sample per difficulty"
    )
    p_samples.add_argument("processed_dir")
    p_samples.add_argument("--out", required=True, help="sample output directory")
    p_samples.add_argument("--splits", default=None, help="reuse an existing splits.json")
    p_samples.add_argument("--train", type=float, default=0.8)
    p_samples.add_argument("--val", type=float, default=0.1)
    p_samples.add_argument("--test", type=float, default=0.1)
    p_samples.add_argument("--seed", type=int, default=0)
    p_samples.add_argument(
        "--require-audio",
        action="store_true",
        help="skip songs that have no audio (recommended for audio-to-chart training)",
    )
    p_samples.set_defaults(func=_cmd_dataset_samples)

    p_fetch = dsub.add_parser(
        "fetch-wmc",
        help="search and download charts from AWMC-MaiArchive (download.wmc.pub)",
    )
    p_fetch.add_argument("-q", "--query", default="", help="title / artist / alias / id")
    p_fetch.add_argument(
        "--id",
        dest="ids",
        action="append",
        default=None,
        help="download a specific chart id (repeatable)",
    )
    p_fetch.add_argument("--version", default=None, help="version id or name filter")
    p_fetch.add_argument("--type", dest="chart_type", choices=["st", "dx", "utg"])
    p_fetch.add_argument("--genre", default=None)
    p_fetch.add_argument(
        "--sort", choices=["id", "title", "version", "popular"], default="id"
    )
    p_fetch.add_argument("--limit", type=int, default=20, help="max songs (1-500)")
    p_fetch.add_argument("--page", type=int, default=1)
    p_fetch.add_argument("--raw", default="data/raw", help="raw dataset root")
    p_fetch.add_argument("--out", default="data/processed", help="processed root")
    p_fetch.add_argument(
        "--cache", default="out/wmc-cache", help="downloaded zip cache directory"
    )
    p_fetch.add_argument("--align", action="store_true", help="run audio alignment")
    p_fetch.add_argument(
        "--include-video", action="store_true", help="download background video too"
    )
    p_fetch.add_argument("--delay", type=float, default=1.0, help="seconds between songs")
    p_fetch.add_argument("--force", action="store_true", help="re-download existing songs")
    p_fetch.add_argument("--dry-run", action="store_true", help="only print matches")
    p_fetch.set_defaults(func=_cmd_dataset_fetch_wmc)

    p_ml = sub.add_parser("ml", help="P3 neural model feature prep, training, inference")
    mlsub = p_ml.add_subparsers(dest="ml_command", required=True)

    p_features = mlsub.add_parser("prepare-features", help="precompute log-mel features")
    p_features.add_argument("processed_dir")
    p_features.add_argument("--out", required=True, help="features directory")
    p_features.add_argument("--sr", type=int, default=22050)
    p_features.add_argument("--n-mels", type=int, default=80)
    p_features.add_argument("--n-fft", type=int, default=1024)
    p_features.add_argument("--hop", type=int, default=256)
    p_features.add_argument("--overwrite", action="store_true")
    p_features.add_argument("--limit", type=int, default=None)
    p_features.add_argument(
        "--songs", default=None, help="comma-separated song ids to extract"
    )
    p_features.set_defaults(func=_cmd_ml_prepare)

    p_prepare_data = mlsub.add_parser(
        "prepare-data",
        help="export training samples and precompute features in one step",
    )
    p_prepare_data.add_argument("processed_dir")
    p_prepare_data.add_argument("--samples-out", default="data/samples")
    p_prepare_data.add_argument("--features-out", default="data/features")
    p_prepare_data.add_argument(
        "--songs", default=None, help="comma-separated song ids (samples and features)"
    )
    p_prepare_data.add_argument("--seed", type=int, default=0)
    p_prepare_data.add_argument("--require-audio", action="store_true")
    p_prepare_data.add_argument("--overwrite-features", action="store_true")
    p_prepare_data.set_defaults(func=_cmd_ml_prepare_data)

    p_train = mlsub.add_parser("train", help="train the P3 event model")
    p_train.add_argument("--config", default="configs/p3_base.yaml")
    p_train.add_argument("--output-dir", default=None)
    p_train.add_argument("--resume", default=None)
    p_train.add_argument(
        "--init-from",
        default=None,
        help=(
            "initialize model weights from a baseline checkpoint, keeping "
            "the structured-token projections at zero"
        ),
    )
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--num-workers", type=int, default=None)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.add_argument("--limit-batches", type=int, default=None)
    p_train.add_argument("--log-every", type=int, default=None)
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=_cmd_ml_train)

    p_infer = mlsub.add_parser("infer", help="generate a chart with a checkpoint")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--audio", required=True)
    p_infer.add_argument("-o", "--output", required=True, help="maidata.txt output")
    p_infer.add_argument("--difficulty", default="master")
    p_infer.add_argument("--level", type=float, default=None)
    p_infer.add_argument("--timing", default=None)
    p_infer.add_argument("--features", default=None, help="precomputed .npz features")
    p_infer.add_argument("--device", default=None)
    p_infer.add_argument("--window-bars", type=int, default=None)
    p_infer.add_argument("--stride-bars", type=int, default=None)
    p_infer.add_argument("--context-bars", type=int, default=None)
    p_infer.add_argument("--max-new-tokens", type=int, default=2048)
    p_infer.add_argument("--temperature", type=float, default=0.8)
    p_infer.add_argument("--top-k", type=int, default=20)
    p_infer.add_argument("--seed", type=int, default=0)
    p_infer.add_argument("--bpm", type=float, default=None, help="override BPM")
    p_infer.add_argument("--first", type=float, default=None, help="override first beat seconds")
    p_infer.add_argument("--title", default=None)
    p_infer.add_argument("--artist", default=None)
    p_infer.set_defaults(func=_cmd_ml_infer)

    p_web = sub.add_parser("web", help="launch the local web UI (browse/infer/train)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--no-browser", action="store_true")
    p_web.set_defaults(func=_cmd_web)

    p_demo = sub.add_parser("demo", help="run synthetic click-track demo")
    p_demo.add_argument("--out", default="out/demo")
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _load_any(path: str) -> Maidata:
    raw = Path(path).read_text(encoding="utf-8")
    if raw.lstrip().startswith("{"):
        return Maidata.from_dict(json.loads(raw))
    return parse_file(path)


def _cmd_parse(args) -> int:
    maidata = parse_file(args.maidata)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(maidata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote IR: {out}")
    else:
        print(f"title: {maidata.get('title', '')}")
        print(f"artist: {maidata.get('artist', '')}")
        print(f"charts: {len(maidata.charts)}")
        for chart in maidata.charts:
            print(
                f"  inote_{chart.index}: level={chart.level or '-'} "
                f"notes={chart.note_count} bpm={chart.initial_bpm:g} "
                f"end_tick={chart.end_tick}"
            )
    for warning in maidata.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_compile(args) -> int:
    maidata = _load_any(args.ir)
    text = compile_maidata(maidata)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote maidata: {out} ({len(text)} bytes)")
    return 0


def _cmd_validate(args) -> int:
    maidata = _load_any(args.input)
    report = validate_maidata(maidata)
    if args.json:
        print(
            json.dumps(
                {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if report.ok:
            print("OK: no validation errors")
        else:
            for error in report.errors:
                print(f"error: {error}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        print(
            f"summary: {len(report.errors)} error(s), "
            f"{len(report.warnings)} warning(s)"
        )
    return 0 if report.ok else 1


def _cmd_stats(args) -> int:
    maidata = _load_any(args.input)
    data = maidata_stats(maidata)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(f"{data['title']} / {data['artist']}")
    for chart in data["charts"]:
        print(
            f"inote_{chart['index']} {chart['difficulty']:<8} "
            f"lv={chart['level'] or '-':<4} "
            f"notes={chart['notes_total']:<4} "
            f"n/s={chart['notes_per_second']:<5} "
            f"dur={chart['duration_seconds']:<7} "
            f"max_sim={chart['max_simultaneous']:<2} "
            f"touch={chart['touch_count']:<3} "
            f"div={chart['finest_division']}"
        )
    return 0


def _cmd_roundtrip(args) -> int:
    maidata = parse_file(args.maidata)
    text = compile_maidata(maidata)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote recompiled maidata: {out}")
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        maidata2 = parse_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    ok = maidata.signature() == maidata2.signature()
    if ok:
        print(
            f"OK: round-trip preserved {len(maidata.charts)} chart(s), "
            f"{sum(c.note_count for c in maidata.charts)} event(s)"
        )
        return 0
    print("FAIL: semantic IR changed after round-trip", file=sys.stderr)
    for a, b in zip(maidata.charts, maidata2.charts):
        if a.signature() != b.signature():
            print(f"  first differing chart: inote_{a.index}", file=sys.stderr)
            break
    return 1


def _cmd_eval(args) -> int:
    from .eval.metrics import evaluate_chart

    generated = _load_any(args.generated)
    reference = _load_any(args.reference)
    index = args.index
    if index is None:
        if len(generated.charts) == 1:
            index = generated.charts[0].index
        elif len(reference.charts) == 1:
            index = reference.charts[0].index
        elif generated.chart(5) and reference.chart(5):
            index = 5
        else:
            raise ValueError("multiple charts; pass --index")
    pred = generated.chart(index)
    ref = reference.chart(index)
    if pred is None or ref is None:
        raise ValueError(f"missing inote_{index} in one of the files")
    result = evaluate_chart(pred, ref, tolerance_seconds=args.tolerance)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    counts = result["counts"]
    onset = result["onset"]
    print(
        f"inote_{index}: matched {counts['matched']} / "
        f"ref {counts['reference']} / pred {counts['predicted']}"
    )
    print(
        f"onset F1={onset['f1']:.4f} "
        f"P={onset['precision']:.4f} R={onset['recall']:.4f} "
        f"mean_err={onset['mean_time_error_ms']:.1f}ms"
    )
    print(
        f"position={result['position_accuracy']:.4f} "
        f"type={result['type_accuracy']:.4f} "
        f"exact={result['exact_accuracy']:.4f}"
    )
    if result["duration_mae_ticks"] is not None:
        print(f"duration MAE={result['duration_mae_ticks']:.2f} ticks")
    if result["slide"]["endpoint_accuracy"] is not None:
        print(
            f"slide endpoint={result['slide']['endpoint_accuracy']:.4f} "
            f"({result['slide']['matched']} matched)"
        )
    stats = result["stats"]
    print(
        f"notes/s ref={stats['notes_per_second']['reference']:.3f} "
        f"pred={stats['notes_per_second']['predicted']:.3f} "
        f"max_sim ref={stats['max_simultaneous']['reference']} "
        f"pred={stats['max_simultaneous']['predicted']}"
    )
    return 0


def _cmd_analyze(args) -> int:
    from .audio.features import BPM_LOW_CONFIDENCE, analyze_audio

    timing = analyze_audio(
        args.audio,
        sr=args.sr,
        bpm=args.bpm,
        first=args.first if args.first is not None else 0.0,
    )
    timing.save(args.out)
    print(
        f"bpm={timing.bpm:.3f} (confidence {timing.bpm_confidence:.3f}) "
        f"duration={timing.duration:.2f}s beats={len(timing.beat_times)} "
        f"downbeats={len(timing.downbeat_times)} onsets={len(timing.onset_times)}"
    )
    if timing.bpm_candidates:
        preview = " ".join(
            f"{item['bpm']:.1f}" for item in timing.bpm_candidates[:3]
        )
        print(f"tempo candidates: {preview}")
    if args.bpm is None and timing.bpm_confidence < BPM_LOW_CONFIDENCE:
        print(
            "warning: low BPM confidence; pass --bpm <value> if the beat grid "
            "is wrong"
        )
    print(f"wrote timing: {args.out}")
    return 0


def _cmd_generate(args) -> int:
    from .audio.timing import TimingInfo
    from .generate import RuleConfig, generate_chart, maidata_for_chart

    config = None
    if (
        args.density is not None
        or args.density_scale is not None
        or args.slide_prob is not None
    ):
        config = RuleConfig(
            difficulty=args.difficulty,
            level=args.level,
            seed=args.seed,
            target_density=args.density,
            density_scale=args.density_scale,
            slide_prob=args.slide_prob,
        )
    if args.timing:
        timing = TimingInfo.load(args.timing)
        if args.first is not None:
            timing.first = float(args.first)
        if args.bpm is not None:
            timing.retime(float(args.bpm))
        result = generate_chart(
            timing,
            difficulty=args.difficulty,
            level=args.level,
            seed=args.seed,
            config=config,
            title=args.title,
            artist=args.artist,
        )
    else:
        from .generate import generate_from_audio

        timing, result = generate_from_audio(
            args.audio,
            difficulty=args.difficulty,
            level=args.level,
            seed=args.seed,
            sr=args.sr,
            title=args.title,
            artist=args.artist,
            bpm=args.bpm,
            first=args.first if args.first is not None else 0.0,
            config=config,
        )
    chart = result.chart
    maidata = maidata_for_chart(
        chart,
        title=args.title or timing.tags.get("title") or Path(args.audio).stem,
        artist=args.artist or timing.tags.get("artist") or "",
        wholebpm=timing.bpm,
    )
    text = compile_maidata(maidata)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.ir_out:
        path = Path(args.ir_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(maidata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    stats = chart_stats(chart)
    if args.stats_out:
        path = Path(args.stats_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"bpm={timing.bpm:.3f} duration={timing.duration:.1f}s "
        f"notes={stats['notes_total']} n/s={stats['notes_per_second']:.2f} "
        f"max_sim={stats['max_simultaneous']}"
    )
    print(f"wrote maidata: {out}")
    for warning in result.warnings[:10]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_dataset_ingest(args) -> int:
    from .dataset.ingest import ingest_dataset

    result = ingest_dataset(
        args.raw_dir,
        args.out,
        align=args.align,
        sr=args.sr,
        log=print,
    )
    print(
        f"songs={len(result.records)} warnings={len(result.warnings)} "
        f"errors={len(result.errors)}"
    )
    for error in result.errors[:20]:
        print(f"error: {error}", file=sys.stderr)
    return 1 if result.errors else 0


def _cmd_dataset_report(args) -> int:
    from .dataset.report import report_dataset

    report = report_dataset(args.processed_dir)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote report: {path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.csv:
        rows = _per_chart_rows(Path(args.processed_dir))
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote CSV: {path}")
    return 0


def _cmd_dataset_split(args) -> int:
    from .dataset.split import split_dataset

    result = split_dataset(
        args.processed_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"train={result['counts']['train']} "
        f"val={result['counts']['val']} test={result['counts']['test']}"
    )
    print(f"wrote splits: {path}")
    return 0


def _cmd_dataset_samples(args) -> int:
    from .dataset.samples import export_samples

    summary = export_samples(
        args.processed_dir,
        args.out,
        splits=args.splits,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
        require_audio=args.require_audio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote samples: {Path(args.out).resolve()}")
    return 0


def _cmd_dataset_fetch_wmc(args) -> int:
    from .dataset.wmc import WmcClient
    from .dataset.zipimport import import_zip_archive

    client = WmcClient()
    if args.ids:
        charts = [client.chart(shortid) for shortid in args.ids]
    else:
        if not args.query.strip():
            raise ValueError("请用 --query 或 --id 指定要下载的曲目")
        payload = client.search(
            args.query,
            version=args.version,
            chart_type=args.chart_type,
            genre=args.genre,
            sort=args.sort,
            page=args.page,
            limit=args.limit,
        )
        charts = payload["charts"]
        matched = payload.get("matched", len(charts))
        print(
            f"AWMC-MaiArchive: 匹配 {matched} 首，下载前 {len(charts)} 首"
            f"(第 {payload.get('page', args.page)}/{payload.get('pages', 1)} 页)"
        )
    if not charts:
        print("没有匹配的曲目。")
        return 0

    for chart in charts:
        levels = chart.get("levels") or {}
        level_text = " ".join(
            f"{name}:{levels[key].get('raw', '?')}"
            for key, name in (("2", "BAS"), ("3", "ADV"), ("4", "EXP"), ("5", "MAS"))
            if key in levels
        )
        print(
            f"  {chart.get('shortid')}  {chart.get('title')}  "
            f"[{chart.get('type')}/{chart.get('version')}]  "
            f"BPM {chart.get('bpm')}  {level_text}"
        )
    if args.dry_run:
        return 0

    raw_root = Path(args.raw)
    processed_root = Path(args.out)
    cache = Path(args.cache)
    staging = cache / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for position, chart in enumerate(charts):
        shortid = str(chart.get("shortid") or "")
        song_id = f"wmc_{shortid}"
        if not args.force and (
            (raw_root / song_id).is_dir() or (processed_root / song_id).is_dir()
        ):
            print(f"[{shortid}] 已存在，跳过（--force 可重新下载）")
        else:
            suffix = "_video" if args.include_video else ""
            zip_path = cache / f"{song_id}{suffix}.zip"
            if not (zip_path.is_file() and _zip_ok(zip_path)):
                print(f"[{shortid}] 下载 {chart.get('title')} …")
                client.download_zip(
                    chart, zip_path, include_video=args.include_video
                )
            result = import_zip_archive(
                zip_path,
                raw_root=raw_root,
                processed_root=processed_root,
                filename=f"{song_id}.zip",
                align=args.align,
                staging_root=staging,
            )
            if result["ok"]:
                print(
                    f"[{shortid}] 入库 {result['song_id']}："
                    f"{result['title']} / {result['chart_count']} 个难度"
                )
                for warning in result["warnings"]:
                    print(f"  warning: {warning}")
            else:
                failures += 1
                print(f"[{shortid}] 失败: {'; '.join(result['errors'])}")
        if position + 1 < len(charts) and args.delay > 0:
            time.sleep(args.delay)
    print(f"完成：成功/跳过 {len(charts) - failures}/{len(charts)}")
    return 1 if failures else 0


def _zip_ok(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _per_chart_rows(processed_dir: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(processed_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for chart in meta.get("charts", []):
            alignment = chart.get("alignment") or {}
            rows.append(
                {
                    "song_id": meta.get("song_id"),
                    "title": meta.get("title"),
                    "index": chart.get("index"),
                    "difficulty": chart.get("difficulty"),
                    "level": chart.get("level"),
                    "notes": chart.get("notes_total"),
                    "notes_per_second": chart.get("notes_per_second"),
                    "duration_seconds": chart.get("duration_seconds"),
                    "max_simultaneous": chart.get("max_simultaneous"),
                    "touch_ratio": chart.get("touch_ratio"),
                    "finest_division": chart.get("finest_division"),
                    "alignment_confidence": alignment.get("confidence"),
                    "alignment_offset": alignment.get("best_offset_seconds"),
                }
            )
    return rows


def _cmd_ml_prepare(args) -> int:
    from .ml.prepare import prepare_features

    songs = None
    if args.songs:
        songs = [item.strip() for item in args.songs.split(",") if item.strip()]
        if not songs:
            raise ValueError("--songs 至少要包含一个歌曲 id")
    summary = prepare_features(
        args.processed_dir,
        args.out,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop=args.hop,
        overwrite=args.overwrite,
        limit=args.limit,
        songs=songs,
        log=print,
    )
    print(
        json.dumps(
            {
                "written": summary["written"],
                "skipped": summary["skipped"],
                "errors": summary["errors"],
                "out_dir": summary["out_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not summary["errors"] else 1


def _cmd_ml_prepare_data(args) -> int:
    from .dataset.samples import export_samples
    from .ml.prepare import prepare_features

    songs = None
    if args.songs:
        songs = [item.strip() for item in args.songs.split(",") if item.strip()]
        if not songs:
            raise ValueError("--songs 至少要包含一个歌曲 id")

    print("== export samples ==", flush=True)
    sample_summary = export_samples(
        args.processed_dir,
        args.samples_out,
        seed=args.seed,
        require_audio=args.require_audio,
        songs=songs,
    )
    print(
        json.dumps(
            {
                "samples_out": args.samples_out,
                "rows": sample_summary.get("chart_count"),
                "split_songs": sample_summary.get("split_song_counts"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print("== extract features ==", flush=True)
    feature_summary = prepare_features(
        args.processed_dir,
        args.features_out,
        overwrite=args.overwrite_features,
        songs=songs,
        log=print,
    )
    print(
        json.dumps(
            {
                "written": feature_summary["written"],
                "skipped": feature_summary["skipped"],
                "errors": feature_summary["errors"],
                "out_dir": feature_summary["out_dir"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if not feature_summary["errors"] else 1


def _cmd_ml_train(args) -> int:
    from .ml.train import train

    return int(train(args) or 0)


def _cmd_ml_infer(args) -> int:
    from .ml.infer import infer

    summary = infer(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        difficulty=args.difficulty,
        level=args.level,
        timing_path=args.timing,
        features_path=args.features,
        device_name=args.device,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        context_bars=args.context_bars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        bpm=args.bpm,
        first=args.first,
        title=args.title,
        artist=args.artist,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_web(args) -> int:
    try:
        from .webapp.app import main as web_main
    except ImportError:  # pragma: no cover - optional dependency
        print(
            "FastAPI is required for the web UI. "
            "Install it with: pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 1
    return int(
        web_main(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        or 0
    )


def _cmd_demo(args) -> int:
    from .demo import run_demo

    summary = run_demo(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
