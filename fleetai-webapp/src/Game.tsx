import React from "react";
import ReactGA from "react-ga";
import {allSunk, Coord, Ship} from "./util";
import Board from "./Board";
import "./Game.css";
import BattleshipActor from "./BattleshipActor";

interface GameProps {
    humanShips: Ship[];
    botShips: Ship[];
    humanShots: Coord[];
    botShots: Coord[];
    addHumanShot: (coord: Coord) => Promise<void>;
    addBotShot: (coord: Coord) => Promise<void>;
    reset: () => void;
}

interface GameState {
    humanTurn: boolean;
    humanWon: boolean | null;
}

export default class Game extends React.Component<GameProps, GameState> {
    private actor: BattleshipActor | undefined;
    constructor(props: Readonly<GameProps> | GameProps) {
        super(props);
        this.state = {humanTurn: Math.random() >= 0.5, humanWon: null};
        this.tileClicked = this.tileClicked.bind(this);
        this.doBotMove = this.doBotMove.bind(this);
    }

    componentDidMount() {
        this.actor = new BattleshipActor();
        if (!this.state.humanTurn) {
            this.doBotMove();
        }
    }

    async getBotMove(): Promise<Coord> {
        let actor = this.actor as BattleshipActor;
        return await actor.getAction(this.props.humanShips, this.props.botShots);
    }

    doBotMove() {
        this.getBotMove().then(this.props.addBotShot).then(() => this.setState({humanTurn: true}));
    }

    componentDidUpdate(prevProps: Readonly<GameProps>, prevState: Readonly<GameState>) {
        if (this.state.humanWon === null && prevState.humanTurn !== this.state.humanTurn) {
            // check winner
            let shots = prevState.humanTurn ? this.props.humanShots : this.props.botShots;
            let ships = prevState.humanTurn ? this.props.botShips : this.props.humanShips;
            if (allSunk(ships, shots)) {
                ReactGA.event({
                    category: "Game",
                    action: "Finished game",
                    label: `${prevState.humanTurn ? "Human" : "Bot"} won`,
                    value: prevState.humanTurn ? 1 : 0
                });
                this.setState({humanWon: prevState.humanTurn});
            } else if (!this.state.humanTurn) {
                this.doBotMove();
            }
        }
    }

    tileClicked(coord: Coord, rightClick: boolean) {
        if (this.state.humanTurn && !rightClick && this.state.humanWon === null) {
            if (!this.props.humanShots.some(shot => shot.row === coord.row && shot.col === coord.col)) {
                this.props.addHumanShot(coord).then(() => this.setState({humanTurn: false}))
            }
        }
    }

    render() {
        let info;
        let hasWinner = false;
        if (this.state.humanWon !== null) {
            info = `${this.state.humanWon ? "You" : "Bot"} won!`;
            hasWinner = true;
        } else {
            info = this.state.humanTurn ? "Your turn!" : "Waiting for bot...";
        }
        return (
            <div id="game">
                <div id="info">
                    <p id="info">
                        {info}
                    </p>
                    {this.state.humanWon !== null ? <button onClick={() => this.props.reset()}>New game</button> :
                        <p id="instructions">Click a square to shoot</p>}
                </div>
                <div id="boards">
                    <div id="bot-ships">
                        <p>Bot's Ships</p>
                        <Board ships={this.props.botShips} shots={this.props.humanShots} tileClicked={this.tileClicked}
                            hideShips={!hasWinner}/>
                    </div>
                    <div id="human-ships">
                        <p>Your Ships</p>
                        <Board ships={this.props.humanShips} shots={this.props.botShots} tileClicked={() => undefined}
                            hideShips={false}/>
                    </div>
                </div>
            </div>
        );
    }
}
