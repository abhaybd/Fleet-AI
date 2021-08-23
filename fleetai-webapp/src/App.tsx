import React from 'react';
import './App.css';
import BoardSetup from "./BoardSetup";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <BoardSetup setHumanBoard={() => null}/>
      </header>
    </div>
  );
}

export default App;
