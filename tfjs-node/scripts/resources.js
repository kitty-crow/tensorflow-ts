/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');
const path = require('node:path');
const {pipeline} = require('node:stream/promises');
const {URL} = require('node:url');
const {HttpsProxyAgent} = require('https-proxy-agent');
const tar = require('tar');
const AdmZip = require('adm-zip');

function requestOptions() {
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy ||
      process.env.HTTP_PROXY || process.env.http_proxy;
  return proxy ? {agent: new HttpsProxyAgent(proxy)} : {};
}

function request(uri) {
  const client = uri.startsWith('https:') ? https : http;
  return new Promise((resolve, reject) => {
    const req = client.get(uri, requestOptions(), response => {
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Download failed with HTTP ${response.statusCode}`));
        return;
      }
      resolve(response);
    });
    req.once('error', reject);
  });
}

async function downloadAndUnpackResource(uri, destPath) {
  const parsed = new URL(uri);
  const response = await request(parsed.href);

  if (parsed.pathname.endsWith('.zip')) {
    const tempFileName = path.join(destPath, '_tmp.zip');
    await pipeline(response, fs.createWriteStream(tempFileName));
    try {
      const zip = new AdmZip(tempFileName);
      zip.extractAllTo(destPath, true);
    } finally {
      await fs.promises.rm(tempFileName, {force: true});
    }
    return;
  }

  if (parsed.pathname.endsWith('.tar.gz')) {
    await pipeline(response, tar.x({cwd: destPath, strict: true}));
    return;
  }

  response.resume();
  throw new Error(`Unsupported packed resource: ${uri}`);
}

module.exports = {downloadAndUnpackResource};
