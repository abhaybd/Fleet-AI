import {Tensor, InferenceSession} from "onnxjs";
import {BOARD_SIZE, Coord, Ship} from "./util";

export default class BattleshipActor {
    private session: InferenceSession;
    private loaded: boolean;

    constructor() {
        this.session = new InferenceSession({backendHint: "cpu"});
        this.loaded = false;
    }

    async getAction(ships: Ship[], shots: Coord[]) {
        if (!this.loaded) {
            await this.session.loadModel(process.env.PUBLIC_URL + "/converted_actor.onnx");
            this.loaded = true;
        }
        let hits: (0 | 1)[] = [];
        let misses: (0 | 1)[] = [];
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                let shot = shots.some(s => s.row === row && s.col === col);
                let coord = new Coord(row, col);
                let shipPresent = ships.some(s => s.collides(coord));
                hits.push(shot && shipPresent ? 1 : 0);
                misses.push(shot && !shipPresent ? 1 : 0);
            }
        }

        let shipsSunk =
            ships.map(s => s.coords().every(c1 => hits[c1.row * BOARD_SIZE + c1.col])).map(sunk => sunk ? 1 : 0);
        let obs = misses.concat(hits).concat(shipsSunk);
        let tensor = new Tensor(new Float32Array(obs), "float32", [obs.length]);
        const inputs = [tensor]
        let outputMap = await this.session.run(inputs);
        let probs = outputMap.values().next().value;
        let options = [];
        for (let i = 0; i < probs.size; i++) {
            options.push([i, probs.data[i]]);
        }
        options.sort((a, b) => b[1] - a[1]);
        for (let [idx, prob] of options) {
            if (!hits[idx] && !misses[idx]) {
                let coord = new Coord(Math.floor(idx / BOARD_SIZE), idx % BOARD_SIZE);
                console.log(`Choosing with probability ${prob}: (${coord.row}, ${coord.col})`);
                return coord;
            }
        }
        throw new Error();
    }
}
