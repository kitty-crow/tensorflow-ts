import {parsePackageManifest, parseVersionList, requireBoolean, requireRecord, requireStringArray, requiredEntry} from './release-inputs';

describe('release input validation', () => {
  it('reads dependency maps without losing valid versions', () => {
    const manifest = parsePackageManifest('{"name":"a","version":"1.2.3","dependencies":{"b":"^2.0.0"}}');
    expect(manifest.version).toBe('1.2.3');
    expect(manifest.dependencies).toEqual({b: '^2.0.0'});
    expect(manifest.peerDependencies).toEqual({});
    expect(manifest.private).toBeFalse();
  });

  it('rejects malformed package dependencies and command arguments', () => {
    expect(() => parsePackageManifest('{"dependencies":{"b":12}}')).toThrowError('Expected a string for dependencies.b');
    expect(() => requireBoolean('false', 'dry')).toThrowError('Expected a boolean for dry');
    expect(() => requireStringArray(['a', 1], 'packages')).toThrowError('Expected an array of strings for packages');
    expect(() => requireRecord([], 'arguments')).toThrowError('Expected an object for arguments');
  });

  it('checks published versions and required selections', () => {
    expect(parseVersionList('["1.0.0","2.0.0"]')).toEqual(['1.0.0', '2.0.0']);
    expect(() => parseVersionList('null')).toThrowError('Expected an array of strings for published versions');
    expect(() => requiredEntry([], 0, 'release unit')).toThrowError('Missing release unit at index 0');
  });
});
