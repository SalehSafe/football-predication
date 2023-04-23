import React, { useState, useEffect } from 'react';
import './style.css'

function HomeTeamDropdown(props) {
  const [teamData, setTeamData] = useState({});

  useEffect(() => {
    fetch("http://127.0.0.1:8000/team_code")
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setTeamData(data);
      })
      .catch((error) => console.log(error));
  }, []);
  const handleSelectChange = (event) => {
    const selectedTeam = event.target.value;
    const selectedTeamName = event.target.options[event.target.selectedIndex].text;
    console.log(`Selected Home Team name: ${selectedTeamName}`);
    console.log(`Selected Home Team: ${selectedTeam}`);
  
    props.onSelectTeam(selectedTeam);
    props.onSelectTeamName(selectedTeamName);
  };
  

  return (
    <div className="dropdown">
      <label className='team-lable' htmlFor="home-team-select">Select a Home Team:</label>
      {teamData.teams && (
        <select
          id="home-team-select"
          name="team"
          className="form-control"
          onChange={handleSelectChange}
        >
          <option value="">Choose a team</option>
          {teamData.teams.map((team, index) => (
            <option key={team} value={teamData.ids[index]}>
              {team}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

export default HomeTeamDropdown;
