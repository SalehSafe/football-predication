import React, { useState, useEffect } from 'react';

function AwayTeamDropdown(props) {
  const [awayData, setAwayData] = useState({});

  useEffect(() => {
    fetch("http://127.0.0.1:8000/opp_code")
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setAwayData(data);
      })
      .catch((error) => console.log(error));
  }, []);

  const handleSelectChange = (event) => {
    const selectedTeam = event.target.value;
    console.log(`Selected Away Team: ${selectedTeam}`);
    props.onSelectTeam(selectedTeam);
  };

  return (
    <div className="dropdown">
      <label className='team-lable' htmlFor="away-team-select">Select an Away Team:</label>
      {awayData.opp && (
        <select
          id="away-team-select"
          name="team"
          className="form-control"
          onChange={handleSelectChange}
        >
          <option value="">Choose a team</option>
          {awayData.opp.map((team, index) => (
            <option key={team} value={awayData.id[index]}>
              {team}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

export default AwayTeamDropdown;
