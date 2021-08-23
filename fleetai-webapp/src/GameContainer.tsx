import React from "react"
import {BOARD_SIZE, canPlace, Coord, Ship, SHIP_LENS} from "./util";
import BoardSetup from "./BoardSetup";
import Game from "./Game";

interface GameContainerState {
    humanShips: Ship[] | null;
    botShips: Ship[] | null;
    humanShots: Coord[]; // shots taken by human
    botShots: Coord[]; // shots taken by bot
}

function randomShips() {
    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    let ships: Ship[] = [];
    for (const shipLen of SHIP_LENS.reverse()) {
        let placed = false;
        while (!placed) {
            let row = Math.floor(Math.random() * BOARD_SIZE);
            let col = Math.floor(Math.random() * BOARD_SIZE);
            let [dr, dc] = dirs[Math.floor(Math.random() * dirs.length)];
            let ship = new Ship(shipLen, row, col, dr, dc);
            if (canPlace(ships, ship)) {
                placed = true;
                ships.push(ship);
            }
        }
    }
    return ships;
}

export default class GameContainer extends React.Component<{}, GameContainerState>{
    constructor(props: {} | Readonly<{}>) {
        super(props)
        this.state = {humanShots: [], humanShips: null, botShips: null, botShots: []};
        this.startGame = this.startGame.bind(this);
        this.addBotShot = this.addBotShot.bind(this);
        this.addHumanShot = this.addHumanShot.bind(this);
    }

    startGame(ships: Ship[]) {
        this.setState({humanShips: ships, botShips: randomShips()});
    }

    async addBotShot(coord: Coord) {
        return new Promise<void>(resolve => {
            this.setState(state => ({botShots: state.botShots.concat(coord)}), resolve);
        })
    }

    async addHumanShot(coord: Coord) {
        return new Promise<void>(resolve => {
            this.setState(state => ({humanShots: state.humanShots.concat(coord)}), resolve);
        })
    }

    render() {
        let content;
        if (!this.state.humanShips) {
            content = <BoardSetup setHumanBoard={this.startGame}/>;
        } else {
            content = <Game humanShips={this.state.humanShips} botShips={this.state.botShips as Ship[]}
                            humanShots={this.state.humanShots} botShots={this.state.botShots}
                            addBotShot={this.addBotShot} addHumanShot={this.addHumanShot}/>;
        }
        return (
            <>
                {content}
            </>
        )
    }
}

