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
    }

    inRange(val: number, bound1: number, bound2: number) {
        const lo = Math.min(bound1, bound2);
        const hi = Math.max(bound1, bound2);
        return lo <= val && val <= hi;
    }

    collides(coord: Coord) {
        const {row, col} = coord;
        const rowStart = this.coord.row;
        const rowEnd = rowStart + this.dr * (this.size - 1);
        const colStart = this.coord.col;
        const colEnd = colStart + this.dc * (this.size - 1);
        return this.inRange(row, rowStart, rowEnd) && this.inRange(col, colStart, colEnd);
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
