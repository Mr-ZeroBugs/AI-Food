/**
 * The healthy-plate diagram: the plate is drawn with the meal's real shares (vegetables, carbs,
 * protein, clockwise from 12 o'clock), and thin rim ticks mark where 2:1:1 would put each boundary
 * (half, then a quarter, then a quarter). Reading it: the further a colour edge sits from its tick,
 * the further the plate is from the guideline.
 */
import Svg, { Circle, G, Line, Path } from 'react-native-svg';

import { Groups, PLATE_KEYS } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import type { PlateRatio } from '@/lib/types';

function polar(cx: number, cy: number, r: number, frac: number) {
  const a = frac * 2 * Math.PI - Math.PI / 2;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as const;
}

function wedge(cx: number, cy: number, r: number, from: number, to: number) {
  if (to - from >= 0.9999) {
    return `M ${cx - r} ${cy} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0`;
  }
  const [x0, y0] = polar(cx, cy, r, from);
  const [x1, y1] = polar(cx, cy, r, to);
  const large = to - from > 0.5 ? 1 : 0;
  return `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} Z`;
}

export function PlateChart({ ratio, size = 148 }: { ratio: PlateRatio; size?: number }) {
  const t = useTheme();
  const c = size / 2;
  const rim = c - 2;
  const food = rim - size * 0.07;
  const total = ratio.veg + ratio.carb + ratio.protein || 1;

  const wedges = PLATE_KEYS.reduce<{ k: (typeof PLATE_KEYS)[number]; from: number; to: number }[]>((out, k) => {
    const from = out.length ? out[out.length - 1].to : 0;
    return [...out, { k, from, to: from + ratio[k] / total }];
  }, []).filter((w) => w.to - w.from > 0.004);

  return (
    <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} accessibilityRole="image"
      accessibilityLabel={`ผัก ${Math.round(ratio.veg)}% ข้าว-แป้ง ${Math.round(ratio.carb)}% โปรตีน ${Math.round(ratio.protein)}% ของจาน`}>
      <Circle cx={c} cy={c} r={rim} fill={t.surface} stroke={t.separator} strokeWidth={1} />
      <G>
        {wedges.map((w) => (
          <Path key={w.k} d={wedge(c, c, food, w.from, w.to)} fill={Groups[w.k].color} stroke={t.surface} strokeWidth={1.5} />
        ))}
      </G>
      {/* 2:1:1 boundaries at 0 (top), 1/2 and 3/4, drawn across the rim. */}
      {[0, 0.5, 0.75].map((f) => {
        const [x0, y0] = polar(c, c, food - 3, f);
        const [x1, y1] = polar(c, c, rim + 1, f);
        return <Line key={f} x1={x0} y1={y0} x2={x1} y2={y1} stroke={t.text} strokeWidth={size < 64 ? 1 : 2} strokeLinecap="round" />;
      })}
    </Svg>
  );
}
