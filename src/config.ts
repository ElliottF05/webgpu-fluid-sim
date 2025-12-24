export type SimConfig = Readonly<{
    numBodies: number;
    gravConstant: number;
    bhTheta: number;
    substeps: number;
    viewPort: [number, number];
    deltaTime: number;
    epsilonMultiplier: number;
}>;

export function getDefaultSimConfig(canvas: HTMLCanvasElement): SimConfig {
    const numBodies = 250000;
    const gravConstant = 1.0;
    const bhTheta = 0.6;
    const substeps = 1;
    const viewPort: [number, number] = [canvas.width, canvas.height];
    const deltaTime = 0.1 * 1.0 / (60.0 * substeps);
    const epsilonMultiplier = 1.0;

    return {
        numBodies,
        gravConstant,
        bhTheta,
        substeps,
        viewPort,
        deltaTime,
        epsilonMultiplier,
    };
}