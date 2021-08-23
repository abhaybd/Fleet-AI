import React from "react";
import {Coord, Ship} from "./util";
import Board from "./Board";
import "./Game.css";

interface GameProps {
    humanShips: Ship[];
    botShips: Ship[];
    humanShots: Coord[];
    botShots: Coord[];
}

interface GameState {
    humanTurn: boolean;
}

export default class Game extends React.Component<GameProps, GameState> {
    constructor(props: Readonly<GameProps> | GameProps) {
        super(props);
        this.state = {humanTurn: Math.random() >= 0.5};
    }

    tileClicked(coord: Coord, rightClick: boolean) {
        if (this.state.humanTurn) {
            // TODO: do move
        }
    }

    render() {
        return (
            <div id="game">
                <div id="info">

                </div>
                <div id="boards">
                    <div id="bot-ships">
                        <p>Bot's Ships</p>
                        <Board ships={this.props.botShips} shots={this.props.humanShots} tileClicked={this.tileClicked}/>
                    </div>
                    <div id="human-ships">
                        <p>Your Ships</p>
                        <Board ships={this.props.humanShips} shots={this.props.botShots} tileClicked={() => undefined}/>
                    </div>
                </div>
            </div>
        );
    }
}
