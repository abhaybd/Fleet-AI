import BattleshipActor from "./BattleshipActor";
import {allSunk, Coord, Ship} from "./util";

export default async function simulateGame(actor: BattleshipActor, ships: Ship[]) {
    let shots: Coord[] = [];
    do {
        let shot = await actor.getAction(ships, shots);
        shots.push(shot);
    } while (!allSunk(ships, shots));
    return shots.length;
}
