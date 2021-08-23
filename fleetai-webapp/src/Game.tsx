import React from "react";
import {BOARD_SIZE, Coord, Ship, SHIP_LENS} from "./util";
import Board from "./Board";
import "./Game.css";

interface GameProps {
    humanShips: Ship[];
    botShips: Ship[];
    humanShots: Coord[];
    botShots: Coord[];
    addHumanShot: (coord: Coord) => Promise<void>;
    addBotShot: (coord: Coord) => Promise<void>;
}

interface GameState {
    humanTurn: boolean;
    humanWon: boolean | null;
}

export default class Game extends React.Component<GameProps, GameState> {
    constructor(props: Readonly<GameProps> | GameProps) {
        super(props);
        this.state = {humanTurn: false, humanWon: null};
        this.tileClicked = this.tileClicked.bind(this);
        this.doBotMove = this.doBotMove.bind(this);
    }

    componentDidMount() {
        if (!this.state.humanTurn) {
            setTimeout(this.doBotMove, 1000);
        }
    }

    async getBotMove(): Promise<Coord> {
        return new Promise<Coord>(resolve => {
            setTimeout(() => {
                let n = this.props.botShots.length;
                resolve(new Coord(Math.floor(n/BOARD_SIZE),n % BOARD_SIZE));
            }, 1000);
        })
    }

    doBotMove() {
        this.getBotMove().then(this.props.addBotShot).then(() => this.setState({humanTurn: true}));
    }

    componentDidUpdate(prevProps: Readonly<GameProps>, prevState: Readonly<GameState>) {
        if (this.state.humanWon === null && prevState.humanTurn !== this.state.humanTurn) {
            // check winner
            let shots = prevState.humanTurn ? this.props.humanShots : this.props.botShots;
            let ships = prevState.humanTurn ? this.props.botShips : this.props.humanShips;
            let numHits = shots.filter(coord => ships.some(s => s.collides(coord))).length;
            if (numHits === SHIP_LENS.reduce((a,b) => a+b)) {
                this.setState({humanWon: true});
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
        if (this.state.humanWon !== null) {
            info = `${this.state.humanWon ? "You" : "Bot"} won!`;
        } else {
            info = this.state.humanTurn ? "Your turn!" : "Waiting for bot...";
        }
        return (
            <div id="game">
                <div id="info">
                    <p>
                        {info}
                    </p>
                </div>
                <div id="boards">
                    <div id="bot-ships">
                        <p>Bot's Ships</p>
                        <Board ships={this.props.botShips} shots={this.props.humanShots} tileClicked={this.tileClicked}
                            hideShips={true}/>
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
