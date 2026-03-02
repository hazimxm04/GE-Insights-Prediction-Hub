import React, { useEffect, useState } from 'react';
import DashboardLayout from '../../layout/dashboardlayout';
import Sidebar from '../sidebar/sidebar';
import { SimulationParams, SimulationResult, PredictionType } from '../../types';
import BellCurve from './BellCurve';

const API_BASE = 'http://127.0.0.1:8000';

interface YearStats {
  year: string;
  total_seats: number;
  coalition_filter: string;
  wins_by_coalition: Record<string, number>;
  win_rate_by_coalition: Record<string, number>;
}
interface HeadToHead {
  a: { year: string; coalition: string; wins: number; win_rate: number };
  b: { year: string; coalition: string; wins: number; win_rate: number };
  delta_wins: number;
  delta_rate: number;
}
interface FlippedSeat {
  seat_id: string;
  seat_name: string;
  state: string;
  [key: string]: string;
}
interface ModelReport {
  year: string;
  timestamp: string;
  train_size: number;
  test_size: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  cv_mean_accuracy: number;
  cv_std_accuracy: number;
  cv_scores: number[];
  confusion_matrix: [[number, number], [number, number]];
}

interface CompareReport {
  head_to_head: HeadToHead;
  train: YearStats;
  test: YearStats;
  regional_train: Record<string, { total: number; wins: Record<string, number>; focused_wins: number | null }>;
  regional_test:  Record<string, { total: number; wins: Record<string, number>; focused_wins: number | null }>;
  flipped_seats: FlippedSeat[];
  flipped_count: number;
}

export default function Dashboard() {
  const [params, setParams] = useState<SimulationParams>({
    mode: 'predict',
    trainYear: 'GE14',
    testYear: 'GE14',
    predictionType: 'State',
    coalition: 'All',
    coalitionB: 'All',
    baseVoterPercent: 50,
    swingPercent: 0,
    turnoutRate: 80,
    specificValue: ''
  });

  const [result, setResult] = useState<SimulationResult | null>(null);
  const [compareReport, setCompareReport] = useState<CompareReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelReport, setModelReport] = useState<ModelReport | null>(null);
  const [filterLists, setFilterLists] = useState({
    years: [] as string[],
    states: [] as string[],
    regions: [] as string[],
    seats: [] as string[],
    coalitions: [] as string[],
    coalitionsB: [] as string[]
  });

  // Fetch options for trainYear (Dataset A)
  useEffect(() => {
    const fetchMeta = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/options?year=${params.trainYear}`);
        const data = await res.json();
        if (data.error) console.error('Options API error:', data.error);
        setFilterLists(prev => ({
          ...prev,
          years:      data.years      || [],
          states:     data.states     || [],
          regions:    data.regions    || [],
          seats:      data.seats      || [],
          coalitions: ['All', ...(data.coalitions || [])]
        }));
        setParams(prev => ({ ...prev, coalition: 'All' }));
      } catch (e) {
        console.error('Failed to fetch filter options:', e);
      }
    };
    fetchMeta();
  }, [params.trainYear]);

  // Fetch model evaluation report when year changes
  useEffect(() => {
    const fetchModelReport = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/model-report?year=${params.trainYear}`);
        const data = await res.json();
        if (data.status === 'success') setModelReport(data.report as ModelReport);
        else setModelReport(null);
      } catch (e) {
        setModelReport(null);
      }
    };
    fetchModelReport();
  }, [params.trainYear]);

  // Fetch coalitions for testYear (Dataset B) separately
  useEffect(() => {
    const fetchCoalitionsB = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/options?year=${params.testYear}`);
        const data = await res.json();
        setFilterLists(prev => ({
          ...prev,
          coalitionsB: ['All', ...(data.coalitions || [])]
        }));
        setParams(prev => ({ ...prev, coalitionB: 'All' }));
      } catch (e) {
        console.error('Failed to fetch Dataset B coalitions:', e);
      }
    };
    fetchCoalitionsB();
  }, [params.testYear]);

  const fetchExplanation = async (explainData: Record<string, unknown>) => {
    setIsExplaining(true);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000); // 15s timeout
      const res = await fetch(`${API_BASE}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(explainData),
        signal: controller.signal
      });
      clearTimeout(timeout);
      const data = await res.json();
      if (data.status === 'success') {
        setExplanation(data.explanation);
      } else {
        setExplanation(`Analysis unavailable: ${data.details || data.explanation}`);
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') {
        setExplanation('Analysis timed out — check that ANTHROPIC_API_KEY is set in your backend.');
      } else {
        setExplanation('Analysis unavailable — backend may not be running.');
      }
      console.error('Explanation fetch failed:', e);
    } finally {
      setIsExplaining(false);
    }
  };

  const handleRunSimulation = async () => {
    setIsLoading(true);
    setError(null);
    setCompareReport(null);
    setExplanation(null);

    try {
      if (params.mode === 'compare') {
        const scope  = params.predictionType || 'Nationwide';
        const target = params.specificValue  || '';
        const url    = `${API_BASE}/api/compare?train_year=${params.trainYear}&test_year=${params.testYear}&coalition_a=${encodeURIComponent(params.coalition)}&coalition_b=${encodeURIComponent(params.coalitionB)}&scope=${encodeURIComponent(scope)}&target=${encodeURIComponent(target)}`;
        const res  = await fetch(url);
        const data = await res.json();

        if (data.status === 'success') {
          setCompareReport(data.report as CompareReport);
          const hth = data.report.head_to_head;
          const compareResultObj = {
            coalition: `${params.coalition} vs ${params.coalitionB}`,
            predictionType: `${params.trainYear} → ${params.testYear}` as PredictionType,
            result: `${params.trainYear} ${params.coalition}: ${hth.a.wins}  |  ${params.testYear} ${params.coalitionB}: ${hth.b.wins}`,
            probability: data.report.flipped_count
          };
          setResult(compareResultObj);

          fetchExplanation({
            mode: 'compare',
            year: params.trainYear,
            year_b: params.testYear,
            scope: params.predictionType,
            target: params.specificValue || params.predictionType,
            coalition: params.coalition,
            coalition_b: params.coalitionB,
            outcome: compareResultObj.result,
            probability: hth.a.win_rate,
            swing: 0,
            turnout: 80,
            delta_wins: hth.delta_wins,
            delta_rate: hth.delta_rate,
            flipped_count: data.report.flipped_count
          });
        } else {
          setError(data.message || 'Compare failed');
        }
        return;
      }

      // PREDICT MODE
      const isSeat = params.predictionType === 'Seat';
      const body: Record<string, unknown> = {
        election_year: params.trainYear,
        turnout_rate: params.turnoutRate,
        swing: params.swingPercent,
        coalition: params.coalition,
        base_voter_rate: params.baseVoterPercent,
        seat_filter: isSeat ? (params.specificValue || '') : '',
        state_filter: !isSeat ? (params.specificValue || 'Nationwide') : 'Nationwide'
      };

      const res = await fetch(`${API_BASE}/predict-summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const data = await res.json();

      if (data.error) {
        setError(data.details || data.error);
        return;
      }

      const { wins, total_seats, losses, view, coalition } = data.summary;
      const isSingleSeat = total_seats === 1 && data.seats?.length === 1;

      const displayProbability = isSingleSeat
        ? Math.round(data.seats[0].win_probability * 100)
        : (total_seats > 0 ? Math.round((wins / total_seats) * 100) : 0);

      const displayResult = isSingleSeat
        ? (data.seats[0].verdict_num === 1 ? 'WIN' : 'LOSS')
        : `${wins} / ${total_seats} Seats Won`;

      const predictResult = {
        coalition: coalition || params.coalition,
        predictionType: (params.specificValue || params.predictionType) as PredictionType,
        result: displayResult,
        probability: displayProbability
      };
      setResult(predictResult);

      // Fetch LLM explanation after result is ready
      fetchExplanation({
        mode: 'predict',
        year: params.trainYear,
        scope: params.predictionType,
        target: params.specificValue || params.predictionType,
        coalition: params.coalition,
        outcome: displayResult,
        probability: displayProbability,
        swing: params.swingPercent,
        turnout: params.turnoutRate,
      });

    } catch (e) {
      setError('Cannot connect to backend. Make sure it is running on port 8000.');
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <Sidebar params={params} setParams={setParams} onRun={handleRunSimulation} options={filterLists} />
      <main className="flex-1 p-8 bg-gray-50 flex flex-col gap-6 border-l-2 border-black overflow-y-auto">
        <h1 className="text-5xl font-black uppercase italic tracking-tighter">Predictive<br />Analysis</h1>

        {/* Selection banner */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-black uppercase text-gray-400 tracking-widest">Selection:</span>
          <span className="bg-blue-100 text-blue-800 px-3 py-1 text-xs font-black uppercase tracking-widest">
            {result ? String(result.predictionType).toUpperCase() : 'No selection'}
          </span>
          <span className="ml-auto border-4 border-black bg-black text-white px-4 py-2 text-xl font-black">
            {params.trainYear}
          </span>
        </div>

        {/* Result Cards Row */}
        <div className="flex gap-4">
          <div className="flex-1 border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <label className="text-[10px] font-bold uppercase text-gray-400">Outcome</label>
            <div className="text-5xl font-black mt-2">
              {isLoading ? '...' : (result?.result || '—')}
            </div>
          </div>

          <div className="w-52 border-4 border-black p-6 bg-black text-white text-center flex flex-col justify-center">
            <label className="text-[10px] font-bold uppercase block mb-2">
              {params.mode === 'compare'
                ? 'Seats Flipped'
                : params.predictionType === 'Seat'
                  ? 'Win Probability'
                  : 'Win Rate'}
            </label>
            <div className="text-5xl font-black font-mono text-[#b5f40d]">
              {isLoading ? '...' : params.mode === 'compare'
                ? (result?.probability ?? 0)
                : `${result?.probability ?? 0}%`}
            </div>
          </div>

          <div className="w-44 border-4 border-black bg-yellow-400 flex flex-col items-center justify-center text-center p-4 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <div className="text-2xl font-black">{params.trainYear}</div>
            <div className="text-[9px] font-black uppercase underline mt-1">Verified Dataset</div>
          </div>
        </div>

        {error && (
          <div className="border-4 border-red-500 bg-red-50 p-4 text-red-700 font-bold text-sm">
            ⚠ {error}
          </div>
        )}

        {/* LLM Explanation Card */}
        {(explanation || isExplaining) && (
          <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <div className="flex items-center gap-3 mb-3">
              <h2 className="text-sm font-black uppercase">AI Analysis</h2>
              <span className="text-[9px] font-black uppercase bg-black text-white px-2 py-0.5 tracking-widest">
                Claude
              </span>
            </div>
            <div className="border-b-4 border-yellow-400 mb-4 w-40" />
            {isExplaining ? (
              <div className="flex items-center gap-2 text-gray-400">
                <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{animationDelay:'0ms'}} />
                <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{animationDelay:'150ms'}} />
                <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{animationDelay:'300ms'}} />
                <span className="text-xs font-bold uppercase ml-1">Generating analysis...</span>
              </div>
            ) : (
              <p className="text-sm leading-relaxed text-gray-800 font-medium">{explanation}</p>
            )}
          </div>
        )}

        {/* Model Report Card — always visible, shows placeholder if not trained yet */}
        {true && (
          <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-sm font-black uppercase">Model Performance</h2>
              <span className="text-[9px] font-black uppercase bg-black text-white px-2 py-0.5 tracking-widest">
                {modelReport?.year ?? params.trainYear}
              </span>
            </div>
            <div className="border-b-4 border-yellow-400 mb-4 w-40" />

            {!modelReport ? (
              <div className="border-2 border-dashed border-gray-300 p-6 text-center">
                <p className="text-sm font-black uppercase text-gray-400">No Report Yet</p>
                <p className="text-xs text-gray-400 mt-1">
                  Run <span className="font-mono bg-gray-100 px-1">python model_training.py</span> in your backend folder to generate evaluation metrics.
                </p>
              </div>
            ) : (<>

            {/* Key metrics row */}
            <div className="grid grid-cols-4 gap-2 mb-4">
              {[
                { label: 'Accuracy',  value: modelReport.accuracy },
                { label: 'Precision', value: modelReport.precision },
                { label: 'Recall',    value: modelReport.recall },
                { label: 'F1 Score',  value: modelReport.f1_score },
              ].map(({ label, value }) => (
                <div key={label} className="border-2 border-black p-3 text-center">
                  <div className="text-[9px] font-bold uppercase text-gray-400 mb-1">{label}</div>
                  <div className={`text-xl font-black ${value >= 0.8 ? 'text-green-600' : value >= 0.65 ? 'text-yellow-600' : 'text-red-600'}`}>
                    {(value * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>

            {/* Cross-validation + confusion matrix */}
            <div className="grid grid-cols-2 gap-4">

              {/* Cross-validation */}
              <div className="border-2 border-black p-3">
                <div className="text-[9px] font-black uppercase text-gray-400 mb-2">5-Fold Cross Validation</div>
                <div className="text-2xl font-black mb-1">
                  {(modelReport.cv_mean_accuracy * 100).toFixed(1)}%
                  <span className="text-sm font-normal text-gray-400 ml-1">
                    ±{(modelReport.cv_std_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex gap-1 mt-2">
                  {modelReport.cv_scores.map((s, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                      <div
                        className="w-full bg-black"
                        style={{ height: `${Math.round(s * 40)}px` }}
                      />
                      <span className="text-[8px] font-mono text-gray-400">{(s * 100).toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Confusion Matrix */}
              <div className="border-2 border-black p-3">
                <div className="text-[9px] font-black uppercase text-gray-400 mb-2">Confusion Matrix</div>
                <div className="text-[9px] text-gray-400 mb-1 text-center">← Predicted →</div>
                <div className="grid grid-cols-3 gap-0.5 text-center text-[10px]">
                  <div />
                  <div className="font-black uppercase text-gray-500">Loss</div>
                  <div className="font-black uppercase text-gray-500">Win</div>

                  <div className="font-black uppercase text-gray-500 text-right pr-1">Loss</div>
                  <div className="bg-green-100 border-2 border-green-400 p-2 font-black">
                    {modelReport.confusion_matrix[0][0]}
                  </div>
                  <div className="bg-red-100 border-2 border-red-300 p-2 font-black text-red-600">
                    {modelReport.confusion_matrix[0][1]}
                  </div>

                  <div className="font-black uppercase text-gray-500 text-right pr-1">Win</div>
                  <div className="bg-red-100 border-2 border-red-300 p-2 font-black text-red-600">
                    {modelReport.confusion_matrix[1][0]}
                  </div>
                  <div className="bg-green-100 border-2 border-green-400 p-2 font-black">
                    {modelReport.confusion_matrix[1][1]}
                  </div>
                </div>
                <div className="text-[8px] text-gray-400 mt-2">
                  Train: {modelReport.train_size} rows · Test: {modelReport.test_size} rows
                </div>
              </div>
            </div>
            </>)}

          </div>
        )}

        {/* Confidence Distribution — predict mode only */}
        {params.mode === 'predict' && (
          <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <h2 className="text-sm font-black uppercase">Confidence Distribution</h2>
            <div className="border-b-4 border-yellow-400 mb-4 w-40 mt-1" />
            <BellCurve centerPoint={result?.probability ?? 50} />
          </div>
        )}

        {/* ── COMPARE RESULTS ── */}
        {params.mode === 'compare' && compareReport && (
          <>
            {/* Head-to-Head comparison card */}
            <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <h2 className="text-sm font-black uppercase mb-1">Head-to-Head</h2>
              <div className="border-b-4 border-yellow-400 mb-4 w-40" />
              <div className="grid grid-cols-3 gap-3 items-center">

                {/* Dataset A */}
                <div className="border-4 border-black p-4 bg-white text-center">
                  <div className="text-[9px] font-black uppercase text-gray-400 mb-1">{compareReport.head_to_head.a.year}</div>
                  <div className="text-xs font-black uppercase bg-black text-white px-2 py-0.5 inline-block mb-2">
                    {compareReport.head_to_head.a.coalition}
                  </div>
                  <div className="text-4xl font-black">{compareReport.head_to_head.a.wins}</div>
                  <div className="text-[10px] text-gray-500 mt-1">seats won</div>
                  <div className="text-sm font-black text-gray-700 mt-1">{compareReport.head_to_head.a.win_rate}%</div>
                </div>

                {/* Delta */}
                <div className={`border-4 p-4 text-center ${compareReport.head_to_head.delta_wins > 0 ? 'border-green-500 bg-green-50' : compareReport.head_to_head.delta_wins < 0 ? 'border-red-500 bg-red-50' : 'border-gray-300 bg-gray-50'}`}>
                  <div className="text-[9px] font-black uppercase text-gray-400 mb-2">Change</div>
                  <div className={`text-3xl font-black ${compareReport.head_to_head.delta_wins > 0 ? 'text-green-600' : compareReport.head_to_head.delta_wins < 0 ? 'text-red-600' : 'text-gray-500'}`}>
                    {compareReport.head_to_head.delta_wins > 0 ? '+' : ''}{compareReport.head_to_head.delta_wins}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">seats</div>
                  <div className={`text-sm font-black mt-1 ${compareReport.head_to_head.delta_rate > 0 ? 'text-green-600' : compareReport.head_to_head.delta_rate < 0 ? 'text-red-600' : 'text-gray-500'}`}>
                    {compareReport.head_to_head.delta_rate > 0 ? '+' : ''}{compareReport.head_to_head.delta_rate}%
                  </div>
                </div>

                {/* Dataset B */}
                <div className="border-4 border-red-400 p-4 bg-white text-center">
                  <div className="text-[9px] font-black uppercase text-red-400 mb-1">{compareReport.head_to_head.b.year}</div>
                  <div className="text-xs font-black uppercase bg-red-400 text-white px-2 py-0.5 inline-block mb-2">
                    {compareReport.head_to_head.b.coalition}
                  </div>
                  <div className="text-4xl font-black">{compareReport.head_to_head.b.wins}</div>
                  <div className="text-[10px] text-gray-500 mt-1">seats won</div>
                  <div className="text-sm font-black text-gray-700 mt-1">{compareReport.head_to_head.b.win_rate}%</div>
                </div>
              </div>

              {/* All coalitions breakdown below */}
              <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t-2 border-gray-200">
                {[compareReport.train, compareReport.test].map((ys) => (
                  <div key={ys.year}>
                    <div className="text-[9px] font-black uppercase text-gray-400 mb-2">{ys.year} — All Coalitions</div>
                    {Object.entries(ys.wins_by_coalition).sort(([,a],[,b]) => b-a).map(([c, wins]) => {
                      const rate = ys.win_rate_by_coalition[c] ?? 0;
                      const isHighlighted = c === ys.coalition_filter && ys.coalition_filter !== 'All';
                      return (
                        <div key={c} className={`flex justify-between text-[10px] px-1 py-0.5 ${isHighlighted ? 'bg-yellow-100 font-black' : ''}`}>
                          <span className="uppercase">{c}</span>
                          <span className="font-mono">{wins} ({rate}%)</span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Regional breakdown side-by-side */}
            <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <h2 className="text-sm font-black uppercase mb-1">Regional Breakdown</h2>
              <div className="border-b-4 border-yellow-400 mb-4 w-40" />
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: params.trainYear, data: compareReport.regional_train },
                  { label: params.testYear,  data: compareReport.regional_test  }
                ].map(({ label, data }) => (
                  <div key={label} className="space-y-1">
                    <div className="text-[10px] font-black bg-black text-white px-2 py-1 inline-block mb-1">{label}</div>
                    {Object.entries(data).map(([region, info]) => {
                      const topCoalition = Object.entries(info.wins).sort(([,a],[,b]) => b - a)[0];
                      return (
                        <div key={region} className="flex justify-between border border-gray-200 px-2 py-1 text-[10px]">
                          <span className="font-bold uppercase truncate mr-2">{region}</span>
                          <span className="font-black text-right shrink-0">
                            {topCoalition ? `${topCoalition[0]}: ${topCoalition[1]}/${info.total}` : '—'}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Seats that changed hands */}
            {compareReport.flipped_seats.length > 0 && (
              <div className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                <h2 className="text-sm font-black uppercase mb-1">
                  Seats That Changed Hands
                  <span className="ml-2 bg-red-500 text-white text-[9px] px-2 py-0.5">{compareReport.flipped_count}</span>
                </h2>
                <div className="border-b-4 border-yellow-400 mb-4 w-40" />
                <div className="space-y-1 max-h-60 overflow-y-auto">
                  {compareReport.flipped_seats.map((seat) => {
                    const fromKey = `winner_${params.trainYear}`;
                    const toKey   = `winner_${params.testYear}`;
                    return (
                      <div key={seat.seat_id} className="grid grid-cols-4 gap-2 border border-gray-200 px-2 py-1 text-[10px] items-center">
                        <span className="font-black">{seat.seat_id}</span>
                        <span className="truncate text-gray-600">{seat.seat_name}</span>
                        <span className="font-bold text-center">{seat[fromKey]}</span>
                        <span className="font-bold text-red-600 text-right">→ {seat[toKey]}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </DashboardLayout>
  );
}