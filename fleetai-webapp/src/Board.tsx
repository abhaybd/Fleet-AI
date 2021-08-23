import React from "react";
import {BOARD_SIZE, Coord, Direction, opposite, Ship} from "./util";
import GameTile from "./GameTile";
import "./Board.css";

interface BoardProps {
    ships: (Ship | null)[];
    shots: Coord[];
    tileClicked: (coord: Coord, rightClicked: boolean) => void;
    hideShips: boolean | boolean[];
}

export default function Board(props: BoardProps) {
    let board: JSX.Element[][] = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
        let row = [];
        for (let c = 0; c < BOARD_SIZE; c++) {
            const coord = new Coord(r, c);
            let shot = props.shots.some(shot => shot.row === r && shot.col === c);
            row.push(
                <GameTile key={r * BOARD_SIZE + c} coord={coord} clicked={props.tileClicked} isOccupied={false} isEdge={false}
                          direction={Direction.East} hitIndicator={shot ? "miss" : undefined}/>
            );
        }
        board.push(row);
    }

    props.ships.forEach((ship, idx) => {
        if (ship) {
            let dir;
            if (ship.dc < 0) dir = Direction.West;
            else if (ship.dc > 0) dir = Direction.East;
            else if (ship.dr < 0) dir = Direction.North;
            else if (ship.dr > 0) dir = Direction.South;
            else throw new Error();

            for (let i = 0; i < ship.size; i++) {
                const row = ship.coord.row + i * ship.dr;
                const col = ship.coord.col + i * ship.dc;
                const isEdge = i === 0 || i === ship.size - 1;
                const d = i !== ship.size - 1 ? dir : opposite(dir);
                let shot = props.shots.some(shot => shot.row === row && shot.col === col);
                let occupied;
                if (typeof props.hideShips === "boolean") occupied = !props.hideShips;
                else occupied = !props.hideShips[idx];
                board[row][col] = <GameTile coord={new Coord(row, col)} clicked={props.tileClicked}
                                            isOccupied={occupied}
                                            isEdge={isEdge} direction={d} key={row * BOARD_SIZE + col}
                                            hitIndicator={shot ? "hit" : undefined}/>;
            }
        }
    });
    return (
        <div id="board">
            {board.map((row, r) => (
                <div className="board-row" key={r}>
                    {row}
                </div>
            ))}
        </div>
    );
}
