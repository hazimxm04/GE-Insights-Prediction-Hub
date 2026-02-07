export type PredictionType = 'Seat' | 'State' | 'Region' | 'Nationwide';

export interface SimulationParams {
  predictionType: PredictionType;
  coalition: string;
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
