import React from 'react';
import { SimulationParams, PredictionType } from '../../types';

interface SidebarProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  onRun: () => void;
  options: { states: string[]; regions: string[]; seats: string[] };
}

const Sidebar: React.FC<SidebarProps> = ({ params, setParams, onRun, options }) => {
  const predictionOptions: PredictionType[] = ['Seat', 'State', 'Region', 'Nationwide'];

  return (
    <aside className="w-80 bg-white p-6 border-r-2 border-black flex flex-col gap-6">
      {/* Prediction Type Dropdown (Simplified) */}
<div>
  <div className="border-2 border-black p-2 flex justify-between items-center bg-white">
    <span className="font-bold">Prediction Type</span>
    <select 
      value={params.predictionType}
      onChange={(e) => setParams({...params, predictionType: e.target.value as PredictionType})}
      className="outline-none bg-transparent cursor-pointer font-bold text-right"
    >
      {predictionOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
    </select>
  </div>
  {/* The extra list box that was here has been removed! */}
</div>
      
      {params.predictionType !== 'Nationwide' && (
        <div className="flex flex-col gap-1">
          <label className="font-bold text-xs uppercase text-gray-500">
            Select Specific {params.predictionType}
          </label>
          <select 
            className="border-2 border-black p-2 bg-white text-sm focus:ring-2 focus:ring-black outline-none"
            value={params.specificValue}
            onChange={(e) => setParams({...params, specificValue: e.target.value})}
          >
            <option value="">-- Choose {params.predictionType} --</option>
            {params.predictionType === 'Seat' && options.seats.map(s => <option key={s} value={s}>{s}</option>)}
            {params.predictionType === 'State' && options.states.map(s => <option key={s} value={s}>{s}</option>)}
            {params.predictionType === 'Region' && options.regions.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
      )}

      {/* Coalition Select */}
      <div className="flex flex-col gap-1">
        <label className="font-bold">Coalition</label>
        <select 
          className="border-2 border-black p-2 bg-white"
          value={params.coalition}
          onChange={(e) => setParams({...params, coalition: e.target.value})}
        >
          <option value="BN">BN</option>
          <option value="Harapan">Harapan</option>
          <option value="PAS">PAS</option>
        </select>
      </div>

      {/* Base Voter Slider */}
    <div className="space-y-2">
        <div className="flex justify-between items-center">
            <label className="font-bold">Base voter %</label>
            {/* This span right here shows the live number */}
            <span className="font-mono bg-black text-white px-2 py-0.5 rounded text-sm">
            {params.baseVoterPercent}%
            </span>
        </div>
        <input 
            type="range" 
            min="0" 
            max="100" 
            value={params.baseVoterPercent}
            onChange={(e) => setParams({...params, baseVoterPercent: parseInt(e.target.value)})}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
    </div>
    
    {/* Swing Slider */}
    <div className="space-y-2">
        <div className="flex justify-between items-center">
            <label className="font-bold">Swing %</label>
            <span className="font-mono bg-black text-white px-2 py-0.5 rounded text-sm">
                {params.swingPercent}%
            </span>
        </div>
        <input 
            type="range" 
            min="-20" 
            max="20" 
            value={params.swingPercent}
            onChange={(e) => setParams({...params, swingPercent: parseInt(e.target.value)})}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
    </div>
      
    
      <button 
        onClick={onRun}
        className="mt-auto border-2 border-black bg-black text-white rounded-full py-2 font-bold hover:bg-white hover:text-black transition-all"
      >
        Run Simulation
      </button>
    </aside>
  );
};

export default Sidebar;