import React from "react";
import "./GameTile.css";
import {Coord, Direction} from "./util";

interface GameTileProps {
    coord: Coord;
    clicked: (coord: Coord, rightClicked: boolean) => void;
    isOccupied: boolean;
    isEdge: boolean;
    direction: Direction;
}

export default class GameTile extends React.Component<GameTileProps, {}> {
    constructor(props: GameTileProps) {
        super(props);
        this.clicked = this.clicked.bind(this);
    }

    clicked(e: React.MouseEvent<HTMLDivElement>) {
        e.preventDefault();
        if (e.button === 0) {
            this.props.clicked(this.props.coord, false);
        } else if (e.button === 2) {
            this.props.clicked(this.props.coord, true);
        }
    }

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
            <div className="game-tile" onClick={this.clicked} onContextMenu={this.clicked}>
                {fill}
            </div>
        );
    }
}
