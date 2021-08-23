import React from "react";
import {BOARD_SIZE, canPlace, Coord, Direction, inRange, Ship, SHIP_LENS} from "./util";
import "./BoardSetup.css";
import Board from "./Board";

interface ShipSelectorProps {
    toPlace: number;
    setToPlace: (idx: number) => void;
    shipDir: Direction;
    setShipDir: (dir: Direction) => void;
    placed: boolean[];
    clear: () => void;
}

function ShipSelector(props: ShipSelectorProps) {
    let instructions;
    if (props.toPlace !== -1) {
        instructions = (
            <>
                <p>
                    Adjust the direction of the ship and click on the board to place.
                </p>
                <p>Right click to delete a ship</p>
            </>
        )
    } else {
        instructions = (
            <>
                <p>Select a ship to place</p>
                <p>Right click to delete a ship</p>
            </>
        )
    }

    function buttonText(idx: number) {
        if (props.placed[idx]) return "Placed!"
        else if (props.toPlace === idx) return "Currently placing...";
        else return "Click to place"
    }
    return (
        <div id="ship-selector">
            <div id="ship-selector-ships">
                {SHIP_LENS.map((len, i) => (
                    <div key={i} className="ship-selector-ship">
                        <div>Ship Length: {len}</div>
                        <div>
                            <button onClick={() => props.setToPlace(props.toPlace === i ? -1 : i)}>
                                {buttonText(i)}
                            </button>
                        </div>
                    </div>
                ))}
            </div>
            <div>
                <button onClick={props.clear}>Clear all</button>
                <select value={props.shipDir} onChange={e => props.setShipDir(e.target.value as Direction)}>
                    {["N", "S", "E", "W"].map(d => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>
            <div id="ship-selector-instructions">
                {instructions}
            </div>
        </div>
    )
}

interface BoardSetupProps {
    setHumanBoard: (ships: Ship[]) => void;
}

interface BoardSetupState {
    ships: (Ship | null)[];
    toPlace: number;
    shipDir: Direction;
}

export default class BoardSetup extends React.Component<BoardSetupProps, BoardSetupState> {
    constructor(props: BoardSetupProps) {
        super(props);
        this.state = {ships: SHIP_LENS.map(() => null), toPlace: -1, shipDir: Direction.East};

        this.tileClicked = this.tileClicked.bind(this);
        this.clear = this.clear.bind(this);
        this.setToPlace = this.setToPlace.bind(this);
    }

    tileClicked(coord: Coord, rightClick: boolean) {
        if (rightClick) {
            this.setState(s => {
                const ships = [...s.ships];
                ships.forEach((ship, i) => {
                    if (ship?.collides(coord)) {
                        ships[i] = null;
                    }
                })
                return {ships: ships};
            });
        } else {
            if (this.state.toPlace !== -1) {
                this.setState(s => {
                    const size = SHIP_LENS[s.toPlace];
                    let dr, dc;
                    if (this.state.shipDir === Direction.East) [dr, dc] = [0, 1];
                    else if (this.state.shipDir === Direction.West) [dr, dc] = [0, -1];
                    else if (this.state.shipDir === Direction.North) [dr, dc] = [-1, 0];
                    else if (this.state.shipDir === Direction.South) [dr, dc] = [1, 0];
                    else throw Error();

                    const ship = new Ship(size, coord.row, coord.col, dr, dc);
                    if (canPlace(s.ships.filter(s => !!s) as Ship[], ship)) {
                        const ships = [...s.ships];
                        ships[s.toPlace] = ship;
                        return {ships: ships, toPlace: -1};
                    } else {
                        return null;
                    }
                });
            }
        }
    }

    clear() {
        this.setState({ships: SHIP_LENS.map(() => null), toPlace: -1});
    }

    setToPlace(idx: number) {
        this.setState(state => {
            if (idx !== -1) {
                const ships = [...state.ships];
                ships[idx] = null;
                return {ships: ships, toPlace: idx};
            } else {
                return {toPlace: idx, ships: state.ships};
            }
        });
    }

    render() {
        return (
            <div id="board-setup">
                <ShipSelector toPlace={this.state.toPlace} setToPlace={this.setToPlace}
                    shipDir={this.state.shipDir} setShipDir={dir => this.setState({shipDir: dir})}
                    placed={this.state.ships.map(s => !!s)} clear={this.clear}/>
                <Board ships={this.state.ships} shots={[]} tileClicked={this.tileClicked} hideShips={false}/>
                <div>
                    <button onClick={() => this.props.setHumanBoard(this.state.ships as Ship[])}
                        disabled={this.state.ships.some(s => !s) ? true: undefined}>
                        Start game
                    </button>
                </div>
            </div>
        )
    }
}
