import React, { useState } from "react";
import HomeTeamDropdown from "./Component/HomeTeamDropdown";
import OpponentDropdown from "./Component/AwayTeamDropdown";
import PredictionComponent from "./Component/Predict";
import PredictRfb from "./Component/PredictRfb";
import PredictKnn from "./Component/predictKnn";
import PredictSvm from "./Component/predictSvm";
import logo from "./assets/logo.png"

import "./App.css";
function App() {
  const [selectedHomeTeam, setSelectedHomeTeam] = useState("");
  const [selectedHomeTeamName, setSelectedHomeTeamName] = useState("");
  const [selectedAwayTeam, setSelectedAwayTeam] = useState("");

  console.log(selectedAwayTeam);

  return (
    <>
    
      <div className="card">
        <HomeTeamDropdown
          onSelectTeam={(team) => setSelectedHomeTeam(team)}
          onSelectTeamName={(team) => setSelectedHomeTeamName(team)}
        />
        <OpponentDropdown onSelectTeam={(team) => setSelectedAwayTeam(team)} />

        <PredictionComponent
          homeTeam={selectedHomeTeam}
          awayTeam={selectedAwayTeam}
          teamName={selectedHomeTeamName}        />

        <PredictKnn
          homeTeam={selectedHomeTeam}
          awayTeam={selectedAwayTeam}
          teamName={selectedHomeTeamName}
        />
        <PredictSvm
          homeTeam={selectedHomeTeam}
          awayTeam={selectedAwayTeam}
          teamName={selectedHomeTeamName}        />
        <PredictRfb
          homeTeam={selectedHomeTeam}
          awayTeam={selectedAwayTeam}
          teamName={selectedHomeTeamName}        />
          <img src={logo} className="logo" ></img>
      </div>
    </>
  );
}

export default App;
