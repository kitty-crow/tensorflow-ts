/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {IOHandler, LoadingIOHandler, LoadOptions, SavingIOHandler} from './types';

export type IORouter = (url: string|string[], loadOptions?: LoadOptions) =>
    IOHandler;

function hasSaveCapability(handler: IOHandler): handler is SavingIOHandler {
  return typeof handler.save === 'function';
}

function hasLoadCapability(handler: IOHandler): handler is LoadingIOHandler {
  return typeof handler.load === 'function';
}

export class IORouterRegistry {
  // Singleton instance.
  private static instance: IORouterRegistry;

  private saveRouters: IORouter[];
  private loadRouters: IORouter[];

  private constructor() {
    this.saveRouters = [];
    this.loadRouters = [];
  }

  private static getInstance(): IORouterRegistry {
    if (IORouterRegistry.instance == null) {
      IORouterRegistry.instance = new IORouterRegistry();
    }
    return IORouterRegistry.instance;
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static registerSaveRouter(saveRouter: IORouter) {
    IORouterRegistry.getInstance().saveRouters.push(saveRouter);
  }

  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  static registerLoadRouter(loadRouter: IORouter) {
    IORouterRegistry.getInstance().loadRouters.push(loadRouter);
  }

  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns All matching handlers with a callable save method.
   * @throws Error if a router supplies a handler without that capability.
   */
  static getSaveHandlers(url: string|string[]): SavingIOHandler[] {
    return IORouterRegistry.getHandlers(url, 'save').map(handler => {
      if (!hasSaveCapability(handler)) {
        throw new Error('A registered save router returned a handler without save().');
      }
      return handler;
    });
  }

  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @param loadOptions Optional, custom load options.
   * @returns All valid handlers for `url`, given the currently registered
   *   handler routers.
   */
  static getLoadHandlers(url: string|string[], loadOptions?: LoadOptions):
      LoadingIOHandler[] {
    return IORouterRegistry.getHandlers(url, 'load', loadOptions).map(handler => {
      if (!hasLoadCapability(handler)) {
        throw new Error('A registered load router returned a handler without load().');
      }
      return handler;
    });
  }

  private static getHandlers(
      url: string|string[], handlerType: 'save'|'load',
      loadOptions?: LoadOptions): IOHandler[] {
    const validHandlers: IOHandler[] = [];
    const routers = handlerType === 'load' ?
        IORouterRegistry.getInstance().loadRouters :
        IORouterRegistry.getInstance().saveRouters;
    routers.forEach(router => {
      const handler = router(url, loadOptions);
      if (handler != null) {
        validHandlers.push(handler);
      }
    });
    return validHandlers;
  }
}

export const registerSaveRouter = (loudRouter: IORouter) =>
    IORouterRegistry.registerSaveRouter(loudRouter);
export const registerLoadRouter = (loudRouter: IORouter) =>
    IORouterRegistry.registerLoadRouter(loudRouter);
export const getSaveHandlers = (url: string|string[]) =>
    IORouterRegistry.getSaveHandlers(url);
export const getLoadHandlers =
    (url: string|string[], loadOptions?: LoadOptions) =>
        IORouterRegistry.getLoadHandlers(url, loadOptions);
