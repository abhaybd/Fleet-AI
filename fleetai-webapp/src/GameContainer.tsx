import React from "react"
import {Coord, Ship} from "./util";
import BoardSetup from "./BoardSetup";
import Game from "./Game";

interface GameContainerState {
    humanShips: Ship[] | null;
    botShips: Ship[] | null;
    humanShots: Coord[]; // shots taken by human
    botShots: Coord[]; // shots taken by bot
}

export default class GameContainer extends React.Component<{}, GameContainerState>{
    constructor(props: {} | Readonly<{}>) {
        super(props)
        this.state = {humanShots: [], humanShips: null, botShips: null, botShots: []};
        this.startGame = this.startGame.bind(this);
    }

    startGame(ships: Ship[]) {
        // TODO: init opponent board (not as the same as human, obv)
        this.setState({humanShips: ships, botShips: ships});
    }

    render() {
        let content;
        if (!this.state.humanShips) {
            content = <BoardSetup setHumanBoard={this.startGame}/>;
        } else {
            content = <Game humanShips={this.state.humanShips} botShips={this.state.botShips as Ship[]}
                            humanShots={this.state.humanShots} botShots={this.state.botShots}/>;
        }
        return (
            <>
                {content}
            </>
        )
    }
}

