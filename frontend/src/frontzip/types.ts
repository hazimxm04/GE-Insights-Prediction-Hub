export type PredictionType = 'Seat' | 'State' | 'Region' | 'Nationwide' | string;
export type SimulationMode = 'predict' | 'compare';

export interface SimulationParams {
  mode: SimulationMode;
  trainYear: string;
  testYear: string;
  predictionType: string;
  coalition: string;    // used in predict mode AND as coalitionA in compare
  coalitionB: string;   // used in compare mode for Dataset B
  baseVoterPercent: number;
  swingPercent: number;
  turnoutRate: number;
  specificValue: string;
}

export interface SimulationResult {
  coalition: string;
  predictionType: PredictionType;
  result: string;
  probability: number;
}