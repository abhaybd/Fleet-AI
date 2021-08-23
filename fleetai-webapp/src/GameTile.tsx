import React from "react";
import "./GameTile.css";
import {Coord, Direction} from "./util";

interface GameTileProps {
    coord: Coord;
    clicked: (coord: Coord, rightClicked: boolean) => void;
    isOccupied: boolean;
    isEdge: boolean;
    direction: Direction;
    hitIndicator: "hit" | "miss" | undefined;
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
        let classNames = [];
        if (this.props.isOccupied) {
            classNames.push("ship-tile");
            if (this.props.isEdge) {
                classNames.push(`ship-tile-${this.props.direction}`);
            }
        }
        let fill = classNames.length ? <div className={classNames.join(" ")} /> : null;
        let hitIndicator = this.props.hitIndicator ? <div className={`shot ${this.props.hitIndicator}`}/> : null;
        return (
            <div className="game-tile" onClick={this.clicked} onContextMenu={this.clicked}>
                {hitIndicator}
                {fill}
            </div>
        );
    }
}
