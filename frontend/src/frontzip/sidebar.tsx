import React from 'react';
import { SimulationParams, PredictionType } from '../../types';

interface SidebarProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  onRun: () => void;
  options: {
    years: string[];
    states: string[];
    regions: string[];
    seats: string[];
    coalitions: string[];   // for Dataset A (trainYear)
    coalitionsB: string[];  // for Dataset B (testYear)
  };
}

const Sidebar: React.FC<SidebarProps> = ({ params, setParams, onRun, options }) => {

  const handleScopeChange = (scope: PredictionType) => {
    setParams(prev => ({ ...prev, predictionType: scope, specificValue: '' }));
  };

  const getTargetOptions = () => {
    if (params.predictionType === 'State')  return options.states;
    if (params.predictionType === 'Region') return options.regions;
    if (params.predictionType === 'Seat')   return options.seats;
    return [];
  };

  const targetOptions = getTargetOptions();

  return (
    <aside className="w-80 bg-white border-r-4 border-black flex flex-col h-full">

      {/* ── Scrollable content ── */}
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">

        {/* 1. Mode */}
        <div className="space-y-1">
          <label className="text-[10px] font-bold uppercase text-gray-500">Operation Mode</label>
          <div className="flex border-2 border-black overflow-hidden">
            {(['predict', 'compare'] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setParams(prev => ({ ...prev, mode }))}
                className={`flex-1 py-2 text-[10px] font-black tracking-widest uppercase ${
                  params.mode === mode ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* 2. Year */}
        <div className="space-y-3">
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-bold uppercase text-gray-500">
              {params.mode === 'compare' ? 'Dataset A (Base Year)' : 'Dataset (GE Year)'}
            </label>
            <select
              value={params.trainYear}
              onChange={(e) => setParams(prev => ({ ...prev, trainYear: e.target.value }))}
              className="border-2 border-black p-2 font-bold text-sm bg-white"
            >
              {(options.years.length > 0 ? options.years : ['GE12', 'GE13', 'GE14']).map(y => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>

          {params.mode === 'compare' && (
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold uppercase text-red-500">Dataset B (Compare Year)</label>
              <select
                value={params.testYear}
                onChange={(e) => setParams(prev => ({ ...prev, testYear: e.target.value }))}
                className="border-2 border-red-500 p-2 font-bold text-sm bg-white text-red-500"
              >
                {(options.years.length > 0 ? options.years : ['GE12', 'GE13', 'GE14']).map(y => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* 2b. Compare Scope + Target (compare mode only) */}
        {params.mode === 'compare' && (
          <div className="space-y-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold uppercase text-gray-500">Analysis Scope</label>
              <div className="grid grid-cols-3 border-2 border-black overflow-hidden">
                {(['Seat', 'State', 'Region'] as const).map(scope => (
                  <button
                    key={scope}
                    onClick={() => setParams(prev => ({ ...prev, predictionType: scope, specificValue: '' }))}
                    className={`py-2 text-[10px] font-black tracking-wider transition-colors ${params.predictionType === scope ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'}`}
                  >
                    {scope.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold uppercase text-gray-500">
                Target {params.predictionType}
              </label>
              <select
                value={params.specificValue}
                onChange={(e) => setParams(prev => ({ ...prev, specificValue: e.target.value }))}
                className="border-2 border-black p-2 font-bold text-xs bg-white"
              >
                <option value="">— Select {params.predictionType} —</option>
                {getTargetOptions().map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {getTargetOptions().length === 0 && (
                <p className="text-[9px] text-gray-400 italic">Loading options...</p>
              )}
            </div>

            {/* Two coalition dropdowns — one per dataset */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold uppercase text-gray-500">Coalition Focus</label>

              {/* Dataset A coalition */}
              <div className="flex flex-col gap-1">
                <label className="text-[9px] font-black uppercase text-gray-400 tracking-widest">
                  Dataset A — {params.trainYear}
                </label>
                <select
                  value={params.coalition}
                  onChange={(e) => setParams(prev => ({ ...prev, coalition: e.target.value }))}
                  className="border-2 border-black p-2 font-bold text-xs bg-white"
                >
                  {(options.coalitions.length > 0 ? options.coalitions : ['All']).map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              {/* Dataset B coalition — needs its own options based on testYear */}
              <div className="flex flex-col gap-1">
                <label className="text-[9px] font-black uppercase text-red-400 tracking-widest">
                  Dataset B — {params.testYear}
                </label>
                <select
                  value={params.coalitionB}
                  onChange={(e) => setParams(prev => ({ ...prev, coalitionB: e.target.value }))}
                  className="border-2 border-red-400 p-2 font-bold text-xs bg-white text-red-600"
                >
                  {(options.coalitionsB.length > 0 ? options.coalitionsB : ['All']).map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* 3. Scope + Target (predict only) */}
        {params.mode === 'predict' && (
          <div className="space-y-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold uppercase text-gray-500">Analysis Scope</label>
              <div className="grid grid-cols-3 border-2 border-black overflow-hidden">
                {(['Seat', 'State', 'Region'] as const).map(scope => (
                  <button
                    key={scope}
                    onClick={() => handleScopeChange(scope)}
                    className={`py-2 text-[10px] font-black tracking-wider transition-colors ${
                      params.predictionType === scope ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'
                    }`}
                  >
                    {scope.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Target dropdown — always visible in predict mode */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold uppercase text-gray-500">
                Target {params.predictionType}
              </label>
              <select
                value={params.specificValue}
                onChange={(e) => setParams(prev => ({ ...prev, specificValue: e.target.value }))}
                className="border-2 border-black p-2 font-bold text-xs bg-white"
              >
                <option value="">— Select {params.predictionType} —</option>
                {targetOptions.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {targetOptions.length === 0 && (
                <p className="text-[9px] text-gray-400 italic">Loading options...</p>
              )}
            </div>
          </div>
        )}

        {/* 4. Coalition (predict only, year-specific) */}
        {params.mode === 'predict' && (
          <div className="space-y-2">
            <label className="text-[10px] font-bold uppercase text-gray-500">
              Coalition <span className="text-gray-400 normal-case font-normal">({params.trainYear})</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(options.coalitions.length > 0 ? options.coalitions : ['All']).map(c => (
                <button
                  key={c}
                  onClick={() => setParams(prev => ({ ...prev, coalition: c }))}
                  className={`border-2 border-black py-2 font-black text-xs transition-all ${
                    params.coalition === c ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'
                  }`}
                >
                  {c.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 5. Sliders (predict only) */}
        {params.mode === 'predict' && (
          <div className="space-y-4 pt-2">
            {[
              { label: 'Swing Rate',      key: 'swingPercent',     min: -20, max: 20,  color: 'black' },
              { label: 'Turnout Rate',    key: 'turnoutRate',      min: 40,  max: 100, color: 'black' },
              { label: 'Base Voter Rate', key: 'baseVoterPercent', min: 0,   max: 100, color: 'blue'  },
            ].map(({ label, key, min, max, color }) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between font-bold text-[10px]">
                  <label className="uppercase">{label}</label>
                  <span className={`px-1 font-mono text-white ${color === 'blue' ? 'bg-blue-500' : 'bg-black'}`}>
                    {params[key as keyof SimulationParams]}%
                  </span>
                </div>
                <input
                  type="range" min={min} max={max}
                  value={params[key as keyof SimulationParams] as number}
                  onChange={(e) => setParams(prev => ({ ...prev, [key]: parseInt(e.target.value) }))}
                  className={`w-full h-1 ${color === 'blue' ? 'accent-blue-500' : 'accent-black'}`}
                />
              </div>
            ))}
          </div>
        )}

      </div>{/* end scrollable */}

      {/* ── Sticky Run Button ── */}
      <div className="p-4 border-t-4 border-black bg-white shrink-0">
        <button
          onClick={onRun}
          className="w-full bg-black text-white py-4 font-black uppercase text-xs tracking-widest border-4 border-black hover:bg-white hover:text-black transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] active:shadow-none active:translate-x-1 active:translate-y-1"
        >
          {params.mode === 'compare' ? '▶ Run Analysis' : '▶ Run Predict'}
        </button>
      </div>

    </aside>
  );
};

export default Sidebar;