export type Config = Readonly<{
    gravConstant: number;
    bhTheta: number;
    substeps: number;
    deltaTime: number;
    epsilonMultiplier: number;
}>;

export function getDefaultConfig(): Config {
    const gravConstant = 1.0;
    const bhTheta = 0.6;
    const substeps = 1;
    const deltaTime = 0.1 * 1.0 / (60.0 * substeps);
    const epsilonMultiplier = 1.0;

    return {
        gravConstant,
        bhTheta,
        substeps,
        deltaTime,
        epsilonMultiplier,
    };
}