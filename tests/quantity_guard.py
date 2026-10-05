"""Conservative, offline quantity consistency check for evaluation only.

Only explicitly typed quantities are compared. Matching equal multisets says
nothing about whether the rest of a translation is correct; role swaps among
equal-valued atoms are inherently invisible to this multiset check.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
import re
import unicodedata


_JA_DIGITS = '零〇一二三四五六七八九'
_WORDS = ('zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
          'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
          'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand')
_NUM = rf'(?:[0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[{_JA_DIGITS}十百千万点]+|(?:{_WORDS})(?:[ -]+(?:{_WORDS}|and))*)'


def _japanese_number(s: str) -> Decimal:
    if '点' in s:
        if s.count('点') != 1:
            raise ValueError
        whole, frac = s.split('点')
        if not frac or any(c not in _JA_DIGITS for c in frac):
            raise ValueError
        digit_values = {'零': 0, '〇': 0, **dict(zip('一二三四五六七八九', range(1, 10)))}
        return Decimal(_japanese_integer(whole or '零')) + Decimal('0.' + ''.join(str(digit_values[c]) for c in frac))
    return Decimal(_japanese_integer(s))


def _japanese_integer(s: str) -> int:
    digits = {'零': 0, '〇': 0, **dict(zip('一二三四五六七八九', range(1, 10)))}
    if not any(c in '十百千万' for c in s):
        return int(''.join(str(digits[c]) for c in s))
    if s.count('万') > 1:
        raise ValueError
    if '万' in s:
        left, right = s.split('万')
        if not left or any(c == '万' for c in right):
            raise ValueError
        return _japanese_submyriad(left, digits) * 10000 + (_japanese_submyriad(right, digits) if right else 0)
    return _japanese_submyriad(s, digits)


def _japanese_submyriad(s: str, digits: dict[str, int]) -> int:
    total = number = 0
    previous_place = 10000
    pending_digit = False
    for c in s:
        if c in digits:
            if pending_digit:
                raise ValueError
            number = digits[c]
            pending_digit = True
        elif c in '十百千':
            place = {'十': 10, '百': 100, '千': 1000}[c]
            if place >= previous_place or (pending_digit and number == 0):
                raise ValueError
            total += (number or 1) * place
            previous_place = place
            number = 0
            pending_digit = False
        else:
            raise ValueError
    return total + number


def _under_hundred(words: list[str], units: dict[str, int], teens: dict[str, int], tens: dict[str, int]) -> int:
    if len(words) == 1:
        word = words[0]
        if word in units:
            return units[word]
        if word in teens:
            return teens[word]
        if word in tens:
            return tens[word]
    if len(words) == 2 and words[0] in tens and words[1] in units and units[words[1]]:
        return tens[words[0]] + units[words[1]]
    raise ValueError


def _under_thousand(words: list[str], units: dict[str, int], teens: dict[str, int], tens: dict[str, int]) -> int:
    if 'hundred' not in words:
        return _under_hundred(words, units, teens, tens)
    if words.count('hundred') != 1:
        raise ValueError
    i = words.index('hundred')
    if i != 1 or words[0] not in units or not units[words[0]]:
        raise ValueError
    return units[words[0]] * 100 + (_under_hundred(words[2:], units, teens, tens) if len(words) > 2 else 0)


def _number(s: str) -> Decimal:
    s = s.casefold().replace(',', '').replace('-', ' ')
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', s):
        return Decimal(s)
    if re.fullmatch(rf'[{_JA_DIGITS}十百千万点]+', s):
        return _japanese_number(s)
    units = dict(zip('zero one two three four five six seven eight nine'.split(), range(10)))
    teens = dict(zip('ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(10, 20)))
    tens = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
    words = re.split(r'[ -]+', s)
    if any(w not in units and w not in teens and w not in tens and w not in ('and', 'hundred', 'thousand') for w in words):
        raise ValueError
    for i, word in enumerate(words):
        if word == 'and' and (i == 0 or words[i - 1] not in ('hundred', 'thousand') or i + 1 == len(words)):
            raise ValueError
    words = [w for w in words if w != 'and']
    if words.count('thousand') > 1:
        raise ValueError
    if 'thousand' in words:
        i = words.index('thousand')
        left = _under_thousand(words[:i], units, teens, tens)
        right = _under_thousand(words[i + 1:], units, teens, tens) if i + 1 < len(words) else 0
        if not 1 <= left <= 999:
            raise ValueError
        value = left * 1000 + right
    else:
        value = _under_thousand(words, units, teens, tens)
    return Decimal(value)


def _canon(v: Decimal) -> str:
    v = v.normalize()
    return format(v, 'f') if v else '0'


def _date_value(month: Decimal, day: Decimal) -> str:
    if month != month.to_integral_value() or day != day.to_integral_value():
        raise ValueError
    value = date(2000, int(month), int(day))  # leap year permits February 29
    return f'{value.month:02d}-{value.day:02d}'


def _time_value(hour: Decimal, minute: Decimal, pm: bool) -> str:
    if hour != hour.to_integral_value() or minute != minute.to_integral_value():
        raise ValueError
    h, m = int(hour), int(minute)
    if not 1 <= h <= 12 or not 0 <= m <= 59:
        raise ValueError
    return f'{(h % 12 + (12 if pm else 0)):02d}:{m:02d}'


def _parse(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    s = unicodedata.normalize('NFKC', text).casefold()
    atoms: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    reasons: list[str] = []

    def add(pattern: str, kind: str, factor: str = '1', date_time=False):
        for m in re.finditer(pattern, s, re.I):
            if any(m.start() < b and m.end() > a for a, b in spans):
                continue
            try:
                if date_time:
                    value = date_time(m)
                else:
                    value = _canon(_number(m.group('n')) * Decimal(factor))
            except (ValueError, InvalidOperation, ArithmeticError):
                reasons.append('unparsed supported-form quantity')
                continue
            atoms.append((kind, value))
            spans.append(m.span())

    # Calendar dates, then AM/PM times. Year-bearing dates are intentionally
    # unsupported because the specified comparison is month/day only.
    months = 'january february march april may june july august september october november december'.split()
    mon = '|'.join(months)
    add(rf'(?P<n>{_NUM})\s*月\s*(?P<day>{_NUM})\s*日', 'date', date_time=lambda m: _date_value(_number(m.group('n')), _number(m.group('day'))))
    add(rf'(?P<month>{mon})\s+(?P<day>{_NUM})(?:st|nd|rd|th)?\b', 'date', date_time=lambda m: _date_value(Decimal(months.index(m.group('month'))+1), _number(m.group('day'))))
    add(rf'(?P<day>{_NUM})\s+(?P<month>{mon})\b', 'date', date_time=lambda m: _date_value(Decimal(months.index(m.group('month'))+1), _number(m.group('day'))))
    add(rf'(?:午前|午後)\s*(?P<n>{_NUM})\s*(?:時|:)\s*(?P<minute>{_NUM})?\s*(?:分)?', 'time', date_time=lambda m: _time_value(_number(m.group('n')), _number(m.group('minute')) if m.group('minute') else Decimal(0), m.group(0).startswith('午後')))
    add(rf'(?P<n>{_NUM})\s*(?::\s*(?P<minute>{_NUM}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)\b', 'time', date_time=lambda m: _time_value(_number(m.group('n')), _number(m.group('minute')) if m.group('minute') else Decimal(0), m.group('ampm').startswith('p')))

    # Compound units are consumed before their component names.
    add(rf'(?P<n>{_NUM})\s*(?:平方\s*キロメートル|平方km|square\s+kilometers?\b|square\s+kilometres?\b|sq\.?\s*km\b|km\s*(?:²|2)(?![a-z0-9]))', 'area', '1000000')
    add(rf'(?P<n>{_NUM})\s*(?:平方\s*メートル|平方m|square\s+meters?\b|square\s+metres?\b|sq\.?\s*m\b|m\s*(?:²|2)(?![a-z0-9]))', 'area')
    add(rf'(?P<n>{_NUM})\s*(?:円|円\b|yen\b|¥)', 'JPY')
    add(rf'(?:¥\s*|JPY\s*)(?P<n>{_NUM})\b', 'JPY')
    add(rf'(?P<n>{_NUM})\s*(?:%|％|percent\b|per\s*cent\b)', 'percent')
    add(rf'(?P<n>{_NUM})\s*(?:km(?![a-z0-9])|kilometers?\b|kilometres?\b|キロメートル)', 'length', '1000')
    add(rf'(?P<n>{_NUM})\s*(?:cm(?![a-z0-9])|centimeters?\b|centimetres?\b|センチメートル)', 'length', '0.01')
    add(rf'(?P<n>{_NUM})\s*(?:m(?![a-z0-9])|meters?\b|metres?\b|メートル)', 'length')
    add(rf'(?P<n>{_NUM})\s*(?:kg(?![a-z0-9])|kilograms?\b|キログラム)', 'mass', '1000')
    add(rf'(?P<n>{_NUM})\s*(?:g(?![a-z0-9])|grams?\b|グラム)', 'mass')
    add(rf'(?P<n>{_NUM})\s*(?:時間|hours?\b|hrs?\b)', 'duration', '60')
    add(rf'(?P<n>{_NUM})\s*(?:分|minutes?\b|mins?\b)', 'duration')

    # ponytail: compound amounts need semantic grouping; abstain instead of
    # treating "1 hour 30 minutes" as two independently comparable amounts.
    ordered = sorted(zip(spans, atoms))
    for ((a, b), left), ((c, d), right) in zip(ordered, ordered[1:]):
        if left[0] == right[0] and s[b:c].strip() in ('', 'and', 'と', '及び', 'および', '+', '、'):
            reasons.append('compound quantity expression requires grouping')

    # Unsupported cues are a hard abstention, even if another quantity parsed.
    kanji_numeral = r'0-9零〇一二三四五六七八九十百千万点'
    ja_range = rf'[{kanji_numeral}][^\s,。]{{0,8}}(?:から|まで|〜|～)[^\s,。]{{0,8}}[{kanji_numeral}]'
    en_range = rf'(?:\d|{_WORDS})[^\s,]{{0,12}}\s+(?:through|to)\s+(?:\d|{_WORDS})'
    if re.search(rf'約|およそ|approximately|\babout\b|\baround\b|between\b|from\b.+\bto\b|{ja_range}|{en_range}|[-−–—]\s*[{kanji_numeral}]|[{kanji_numeral}]\s*[-−–—]\s*[{kanji_numeral}]|minus|negative|マイナス|half|quarter|twice|thrice|double|triple|\b(?:at least|at most|more than|less than|greater than|fewer than|over|under|up to|more|less|fewer|greater|smaller)\b|半|以上|以下|未満|超える|超|時速|毎時|毎分|毎秒|/(?:h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds)\b|\bper\s+(?:hour|minute|second)\b|today|tomorrow|yesterday|next\s+(?:week|month|year)|昨日|今日|明日|来週|来月|来年|正午|真夜中', s):
        reasons.append('approximation, range, relative date/time, or unsupported arithmetic cue')

    residual = list(s)
    for a, b in spans:
        residual[a:b] = ' ' * (b - a)
    rest = ''.join(residual)
    if re.search(r'\d|[零〇一二三四五六七八九十百千万点]', rest):
        reasons.append('unmatched numeric-looking residue')
    if re.search(r'\b(?:yen|JPY|percent|per\s+cent|km|cm|meters?|metres?|kilometers?|kilometres?|kg|grams?|hours?|hrs?|minutes?|mins?|square|sq)\b|円|％|平方|キロ|メートル|センチ|グラム|時間|(?:^|\s)¥', rest):
        reasons.append('unmatched unit cue')
    if re.search(rf'\b(?:{_WORDS})\b', rest, re.I):
        reasons.append('unmatched number word')
    return atoms, list(dict.fromkeys(reasons))


def check(source_ja: str, target_en: str) -> dict:
    """Compare only supported quantity multisets; abstain on ambiguity."""
    src, sr = _parse(source_ja)
    tgt, tr = _parse(target_en)
    result = dict(status='abstain', source_atoms=[dict(type=k, value=v) for k, v in src],
                  target_atoms=[dict(type=k, value=v) for k, v in tgt], reasons=[])
    if sr or tr:
        result['reasons'] = [f'source: {r}' for r in sr] + [f'target: {r}' for r in tr]
    elif not src or not tgt:
        result['reasons'] = ['no comparable quantities in both texts']
    elif {k for k, _ in src} != {k for k, _ in tgt}:
        result['reasons'] = ['typed quantity dimensions do not match']
    elif Counter(k for k, _ in src) != Counter(k for k, _ in tgt):
        result['reasons'] = ['quantity occurrence counts differ; grouping or deduplication may be valid']
    elif Counter(src) != Counter(tgt):
        result['status'] = 'mismatch'
        result['reasons'] = ['typed quantity multisets differ']
    else:
        result['status'] = 'consistent'
        result['reasons'] = ['supported typed quantity multisets match; translation meaning is not assessed']
    return result


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) != 3:
        raise SystemExit('usage: quantity_guard.py SOURCE_JA TARGET_EN')
    print(json.dumps(check(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
