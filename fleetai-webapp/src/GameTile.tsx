import React from "react";
import "./GameTile.css";
import {Coord, Direction} from "./util";

interface GameTileProps {
    coord: Coord;
    clicked: (coord: Coord) => void;
    isOccupied: boolean;
    isEdge: boolean;
    direction: Direction;
}

export default class GameTile extends React.Component<GameTileProps, {}> {
    render() {
        let fill = null;
        if (this.props.isOccupied) {
            if (this.props.isEdge) {
                fill = <div className={`ship-tile ship-tile-${this.props.direction}`}/>
            } else {
                fill = <div className="ship-tile"/>
            }
        }
        return (
            <div className="game-tile" onClick={() => this.props.clicked(this.props.coord)}>
                {fill}
            </div>
        );
    }
}
