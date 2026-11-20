/**
 * Build-time configuration (EAS environment variables, see README):
 *   EXPO_PUBLIC_API_URL  the inference server, e.g. https://kinwhy-api-xxxx.a.run.app
 *   EXPO_PUBLIC_API_KEY  the server's APP_ACCESS_CODE. It ships inside the app, so it only keeps casual
 *                        callers off the server; the server's rate limit is the real cost guard.
 * Without EXPO_PUBLIC_API_URL (development) the app talks to the dev machine and shows server settings.
 */
export const BUILD_API_URL = process.env.EXPO_PUBLIC_API_URL?.replace(/\/+$/, '') || '';
export const BUILD_API_KEY = process.env.EXPO_PUBLIC_API_KEY || '';
export const IS_CONFIGURED_BUILD = BUILD_API_URL.startsWith('https://');
export const PRIVACY_URL = 'https://kinwhy-ai.vercel.app/privacy.html';
