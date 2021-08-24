import React from "react";
import ReactGA from "react-ga";
import {Coord, randomShips, Ship} from "./util";
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
        this.addBotShot = this.addBotShot.bind(this);
        this.addHumanShot = this.addHumanShot.bind(this);
        this.reset = this.reset.bind(this);
    }

    componentDidMount() {
        ReactGA.initialize("UA-174949204-3", {
            titleCase: true,
            gaOptions: {
                siteSpeedSampleRate: 100
            }
        });
        ReactGA.pageview("/");
        ReactGA.event({
            category: "Game",
            action: "New Game"
        });
    }

    startGame(ships: Ship[]) {
        ReactGA.event({
            category: "Game",
            action: "Start Game"
        });
        this.setState({humanShips: ships, botShips: randomShips()});
    }

    reset() {
        ReactGA.event({
            category: "Game",
            action: "Reset Game"
        });
        this.setState({humanShots: [], humanShips: null, botShips: null, botShots: []});
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
                            addBotShot={this.addBotShot} addHumanShot={this.addHumanShot} reset={this.reset}/>;
        }
        return (
            <>
                {content}
            </>
        )
    }
}

