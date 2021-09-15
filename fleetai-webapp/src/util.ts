export const SHIP_LENS = [2, 3, 3, 4, 5];
export const BOARD_SIZE = 10;

export class Coord {
    row: number;
    col: number;

    constructor(row: number, col: number) {
        this.row = row;
        this.col = col;
    }
}

export function allSunk(ships: Ship[], shots: Coord[]) {
    let numHits = shots.filter(coord => ships.some(s => s.collides(coord))).length;
    return numHits === SHIP_LENS.reduce((a,b) => a+b)
}

export function inRange(val: number, bound1: number, bound2: number) {
    const lo = Math.min(bound1, bound2);
    const hi = Math.max(bound1, bound2);
    return lo <= val && val <= hi;
}

export function canPlace(ships: Ship[], ship: Ship) {
    const inBoard = ship.coords().every(
        coord => inRange(coord.row, 0, BOARD_SIZE-1) &&
            inRange(coord.col, 0, BOARD_SIZE-1));
    const collides = ships.some(s => s.collides(ship));
    return inBoard && !collides;
}

export function randomShips() {
    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    let ships: Ship[] = [];
    for (const shipLen of [...SHIP_LENS].reverse()) {
        let placed = false;
        while (!placed) {
            let row = Math.floor(Math.random() * BOARD_SIZE);
            let col = Math.floor(Math.random() * BOARD_SIZE);
            let [dr, dc] = dirs[Math.floor(Math.random() * dirs.length)];
            let ship = new Ship(shipLen, row, col, dr, dc);
            if (canPlace(ships, ship)) {
                placed = true;
                ships.push(ship);
            }
        }
    }
    return ships.reverse();
}

export class Ship {
    coord: Coord;
    size: number;
    dr: number;
    dc: number;

    constructor(size: number, row: number, col: number, dr: number, dc: number) {
        this.size = size;
        this.coord = new Coord(row, col);
        this.dr = dr;
        this.dc = dc;

        this.collides = this.collides.bind(this);
        this.coords = this.coords.bind(this);
    }

    coords() {
        let ret: Coord[] = [];
        let {row, col} = this.coord;
        for (let i = 0; i < this.size; i++) {
            ret.push(new Coord(row + i * this.dr, col + i * this.dc));
        }
        return ret;
    }

    collides(coordOrShip: (Coord | Ship)): boolean {
        if (coordOrShip instanceof Coord) {
            const {row, col} = coordOrShip as Coord;
            const rowStart = this.coord.row;
            const rowEnd = rowStart + this.dr * (this.size - 1);
            const colStart = this.coord.col;
            const colEnd = colStart + this.dc * (this.size - 1);
            return inRange(row, rowStart, rowEnd) && inRange(col, colStart, colEnd);
        } else {
            let coords = coordOrShip.coords();
            return coords.some(this.collides);
        }
    }
}

export enum Direction {
    North = "N", East = "E", South = "S", West = "W"
}

export function opposite(dir: Direction) {
    switch (dir) {
        case Direction.East:
            return Direction.West;
        case Direction.North:
            return Direction.South;
        case Direction.South:
            return Direction.North;
        case Direction.West:
            return Direction.East;
    }
}
