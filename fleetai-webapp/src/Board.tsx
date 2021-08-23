import React from "react";
import {BOARD_SIZE, Coord, Direction, opposite, Ship} from "./util";
import GameTile from "./GameTile";

interface BoardProps {
    ships: (Ship | null)[];
    shots: Coord[];
    tileClicked: (coord: Coord, rightClicked: boolean) => void;
}

export default function Board(props: BoardProps) {
    let board = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
        let row = [];
        for (let c = 0; c < BOARD_SIZE; c++) {
            const coord = new Coord(r, c);
            row.push(
                <GameTile key={r * BOARD_SIZE + c} coord={coord} clicked={props.tileClicked} isOccupied={false} isEdge={false}
                          direction={Direction.East}/>
            );
        }
        board.push(row);
    }

    for (const ship of props.ships) {
        if (ship) {
            let dir;
            if (ship.dc < 0) dir = Direction.West;
            else if (ship.dc > 0) dir = Direction.East;
            else if (ship.dr < 0) dir = Direction.North;
            else if (ship.dr > 0) dir = Direction.South;
            else throw new Error();

            board[ship.coord.row][ship.coord.col] =
                <GameTile coord={ship.coord} clicked={props.tileClicked} isOccupied={true}
                          isEdge={true} direction={dir}/>;
            for (let i = 0; i < ship.size; i++) {
                const row = ship.coord.row + i * ship.dr;
                const col = ship.coord.col + i * ship.dc;
                const isEdge = i === 0 || i === ship.size - 1;
                const d = i !== ship.size - 1 ? dir : opposite(dir);
                board[row][col] = <GameTile coord={new Coord(row, col)} clicked={props.tileClicked} isOccupied={true}
                                            isEdge={isEdge} direction={d} key={row * BOARD_SIZE + col}/>;
            }
        }
    }
    return (
        <div id="board">
            {board.map((row, r) => (
                <div className="game-row" key={r}>
                    {row}
                </div>
            ))}
        </div>
    );
}
