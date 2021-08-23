import React from "react";
import {BOARD_SIZE, Coord, Direction, opposite, Ship, SHIP_LENS} from "./util";
import GameTile from "./GameTile";
import "./Game.css";

interface BoardSetupProps {
    setHumanBoard: (ships: Ship[]) => void;
}

interface BoardSetupState {
    ships: (Ship | null)[];
    toPlace: number;
}

export default class BoardSetup extends React.Component<BoardSetupProps, BoardSetupState> {
    constructor(props: BoardSetupProps) {
        super(props);
        this.state = {ships: SHIP_LENS.map(() => null), toPlace: 0};

        this.tileClicked = this.tileClicked.bind(this);
    }

    tileClicked(coord: Coord) {
        if (this.state.toPlace < SHIP_LENS.length) {
            this.setState(s => {
                const size = SHIP_LENS[s.toPlace];
                const ship = new Ship(size, coord.row, coord.col, 0, 1);
                const ships = [...s.ships];
                ships[s.toPlace] = ship;
                return {ships: ships, toPlace: s.toPlace+1};
            })
        }

    }

    render() {
        let board = [];
        for (let r = 0; r < BOARD_SIZE; r++) {
            let row = [];
            for (let c = 0; c < BOARD_SIZE; c++) {
                const coord = new Coord(r, c);
                row.push(
                    <GameTile key={r * BOARD_SIZE + c} coord={coord} clicked={this.tileClicked} isOccupied={false} isEdge={false}
                              direction={Direction.East}/>
                );
            }
            board.push(row);
        }

        for (const ship of this.state.ships) {
            if (ship) {
                let dir;
                if (ship.dc < 0) dir = Direction.West;
                else if (ship.dc > 0) dir = Direction.East;
                else if (ship.dr < 0) dir = Direction.North;
                else if (ship.dr > 0) dir = Direction.South;
                else throw new Error();

                board[ship.coord.row][ship.coord.col] =
                    <GameTile coord={ship.coord} clicked={this.tileClicked} isOccupied={true}
                              isEdge={true} direction={dir}/>;
                for (let i = 0; i < ship.size; i++) {
                    const row = ship.coord.row + i * ship.dr;
                    const col = ship.coord.col + i * ship.dc;
                    const isEdge = i === 0 || i === ship.size - 1;
                    const d = i !== ship.size - 1 ? dir : opposite(dir);
                    board[row][col] = <GameTile coord={new Coord(row, col)} clicked={this.tileClicked} isOccupied={true}
                                  isEdge={isEdge} direction={d} key={row * BOARD_SIZE + col}/>;
                }
            }
        }
        return (
            <div id="board-setup">
                {board.map(row => (
                    <div className="game-row">
                        {row}
                    </div>
                ))}
            </div>
        );
    }
}
