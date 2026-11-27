// Copyright 2026. Licensed under the Apache License, Version 2.0.

/** Validate untyped JSON and argparse values before release or build actions. */
export function requireRecord(value: unknown, description: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Expected an object for ${description}`);
  }
  return value as Record<string, unknown>;
}

export function requireString(value: unknown, description: string): string {
  if (typeof value !== 'string') {
    throw new Error(`Expected a string for ${description}`);
  }
  return value;
}

export function requireBoolean(value: unknown, description: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`Expected a boolean for ${description}`);
  }
  return value;
}

export function optionalString(value: unknown, description: string): string|null {
  if (value == null) {
    return null;
  }
  return requireString(value, description);
}

export function requireStringArray(value: unknown, description: string): string[] {
  if (!Array.isArray(value) || !value.every(item => typeof item === 'string')) {
    throw new Error(`Expected an array of strings for ${description}`);
  }
  return value;
}

export function requiredEntry<T>(values: readonly T[], index: number, description: string): T {
  const value = values[index];
  if (value == null) {
    throw new Error(`Missing ${description} at index ${index}`);
  }
  return value;
}

export interface PackageManifest {
  name: string|null;
  version: string|null;
  private: boolean;
  dependencies: Record<string, string>;
  peerDependencies: Record<string, string>;
  devDependencies: Record<string, string>;
}

function dependencyRecord(value: unknown, description: string): Record<string, string> {
  if (value == null) {
    return {};
  }
  const record = requireRecord(value, description);
  return Object.fromEntries(Object.entries(record).map(([name, version]) =>
      [name, requireString(version, `${description}.${name}`)]));
}

export function parsePackageManifest(text: string): PackageManifest {
  const raw: unknown = JSON.parse(text);
  const record = requireRecord(raw, 'package manifest');
  return {
    name: optionalString(record.name, 'package name'),
    version: optionalString(record.version, 'package version'),
    private: record.private == null ? false : requireBoolean(record.private, 'package private flag'),
    dependencies: dependencyRecord(record.dependencies, 'dependencies'),
    peerDependencies: dependencyRecord(record.peerDependencies, 'peerDependencies'),
    devDependencies: dependencyRecord(record.devDependencies, 'devDependencies')
  };
}

export function parseVersionList(text: string): string[] {
  const raw: unknown = JSON.parse(text);
  return requireStringArray(raw, 'published versions');
}
