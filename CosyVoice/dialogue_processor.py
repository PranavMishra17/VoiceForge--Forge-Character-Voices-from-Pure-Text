import os
import re
import logging
from pathlib import Path
from CosyVoice.cosyvoice_tts import CosyVoiceTTS

def parse_speed(text):
    """Extract [speed:x] from text, return (cleaned_text, speed)"""
    match = re.match(r'\[speed:([0-9.]+)\]\s*(.*)', text)
    if match:
        speed = float(match.group(1))
        cleaned = match.group(2)
        return cleaned, speed
    return text, None

def process_dialogue(script_path, output_dir, speaker_id):
    tts = CosyVoiceTTS()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["Dialogue Processing Report", "========================", f"Script: {script_path}"]
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    report_lines.append(f"Processed: {len(lines)} lines\n")
    for idx, line in enumerate(lines, 1):
        text, speed = parse_speed(line)
        report_lines.append(f"Line {idx}: {text[:50]}...")
        if speed:
            report_lines.append(f"  Speed: {speed}")
        out_path = output_dir / f"line_{idx}"
        try:
            success = tts.synthesize(text, str(out_path), speaker_id=speaker_id, speed=speed)
            if success:
                report_lines.append(f"  Output: {out_path}_output_0.wav")
            else:
                report_lines.append(f"  Output: None")
        except Exception as e:
            logging.error(f"TTS synthesis failed for line {idx}: {e}")
            report_lines.append(f"  Output: None")
    # Save report
    with open(output_dir / "processing_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n🎉 Processed {len(lines)} dialogue lines!")
    print(f"📋 Report saved: {output_dir / 'processing_report.txt'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CosyVoice2 Dialogue Processor')
    parser.add_argument('--script', type=str, required=True, help='Path to dialogue script (txt)')
    parser.add_argument('--output_dir', type=str, default='output/sample_dialogue', help='Output directory')
    parser.add_argument('--speaker_id', type=str, required=True, help='Speaker ID (must be extracted)')
    args = parser.parse_args()
    process_dialogue(args.script, args.output_dir, args.speaker_id)
