/** Shapes returned by POST /api/analyze-image (backend/app.py + backend/nutrition.build_result). */

export type PlateRatio = { veg: number; carb: number; protein: number };

export type AnalyzeResult =
  | {
      is_food: false;
      message_th: string;
      warnings: string[];
      image_size: [number, number];
    }
  | {
      is_food: true;
      plate: {
        food_coverage: number;
        group_pct: { g1_protein: number; g2_carb: number; g3_veg: number; g4_fruit: number };
        plate_211: { ratio: PlateRatio; target: PlateRatio; score: number | null };
        liquid_coverage?: number;
      };
      detected_classes: { class: string; name_th: string; pct_of_image: number }[];
      advice: string[];
      warnings: string[];
      seg: { food_prob: number; model?: string };
      overlay_png_b64?: string;
      image_size: [number, number];
    };

export type Meal = {
  id: string;
  createdAt: number; // ms since epoch
  photoUri: string; // copy kept in the app's document directory
  status: 'analyzing' | 'done' | 'error';
  error?: string;
  result?: AnalyzeResult;
};

export type Settings = {
  serverUrl: string;
  accessCode: string;
  energyLevel: 1600 | 2000 | 2400;
};
