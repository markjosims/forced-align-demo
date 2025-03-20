from typing import Optional, Sequence
from argparse import ArgumentParser
import parselmouth
import tgt
import pandas as pd

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = ArgumentParser()
    parser.add_argument('--wav', '-w')
    parser.add_argument('--textgrid', '-t')
    parser.add_argument('--output', '-o')
    args = parser.parse_args(argv)

    sound = parselmouth.Sound(args.wav)
    textgrid = tgt.io.read_textgrid(args.textgrid)
    formant = sound.to_formant_burg(
        max_number_of_formants=5,
        maximum_formant=4_500,
    )
    intensity = sound.to_intensity()
    word_tier = textgrid.get_tier_by_name('word')
    phone_tier = textgrid.get_tier_by_name('phone')
    data = []
    for t in formant.ts():
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        amp = intensity.get_value(t)
        phone_intervals = phone_tier.get_annotations_by_time(t)
        if not phone_intervals:
            continue
        phone = phone_intervals[0].text
        word = word_tier.get_annotations_by_time(t)[0].text
        data.append({
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'amp': amp,
            'phone': phone,
            'word': word,
            'time': t,
        })
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    return 0

if __name__ == '__main__':
    main()