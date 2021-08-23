import React from "react"
import {Coord, Ship} from "./util";

interface GameContainerState {
    ships: Ship[];
    shot_coords: Coord[];
}

export default class GameContainer extends React.Component<{}, GameContainerState>{
    constructor(props: {} | Readonly<{}>) {
        super(props)
        this.state = {shot_coords: [], ships: []}
    }

    render() {
        return <></>
    }
}

