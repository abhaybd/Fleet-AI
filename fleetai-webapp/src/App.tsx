import React from 'react';
import ReactGA from "react-ga";
import './App.css';
import GameContainer from "./GameContainer";
import GitHubLogo from "./GitHub-Mark-Light-64px.png";

function ForkMe() {
    return (
        <div id="fork-me">
            <ReactGA.OutboundLink eventLabel="GitHub-Link" to="https://github.com/abhaybd/Fleet-AI">
                <img src={GitHubLogo} alt="GitHub Logo" />
                <span>Fork me on GitHub</span>
            </ReactGA.OutboundLink>
        </div>
    )
}

function App() {
  return (
    <div className="App">
      <header className="App-header">
          <GameContainer/>
          <ForkMe />
      </header>
    </div>
  );
}

export default App;
