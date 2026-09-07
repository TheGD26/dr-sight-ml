// Local persistence for screenings — a drop-in stand-in for the Base44
// `entities.Screening` calls the original app used. Everything lives in
// localStorage under one key; images are stored as data URLs.
//
// localStorage is ~5 MB per origin, so large images are downscaled before they
// are saved (see downscaleToDataUrl). This is a demo/single-device store, not a
// shared backend.

const KEY = "dr-sight:screenings";

function readAll() {
  try {
    const raw = localStorage.getItem(KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function writeAll(arr) {
  try {
    localStorage.setItem(KEY, JSON.stringify(arr));
  } catch (err) {
    throw new Error(
      "Could not save the screening locally — browser storage is full. Delete some history and try again.",
    );
  }
}

function newId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export const Screening = {
  /** @returns {Promise<object[]>} newest first */
  async list() {
    return readAll().sort(
      (a, b) => new Date(b.created_date) - new Date(a.created_date),
    );
  },

  async get(id) {
    const found = readAll().find((s) => s.id === id);
    if (!found) throw new Error("Screening not found");
    return found;
  },

  async create(data) {
    const all = readAll();
    const record = {
      id: newId(),
      created_date: new Date().toISOString(),
      notes: "",
      ...data,
    };
    all.push(record);
    writeAll(all);
    return record;
  },

  async update(id, patch) {
    const all = readAll();
    const i = all.findIndex((s) => s.id === id);
    if (i === -1) throw new Error("Screening not found");
    all[i] = { ...all[i], ...patch };
    writeAll(all);
    return all[i];
  },

  async remove(id) {
    writeAll(readAll().filter((s) => s.id !== id));
  },
};

/**
 * Read a File into a downscaled JPEG/PNG data URL so it fits in localStorage.
 * Falls back to the raw data URL if canvas processing fails.
 */
export function downscaleToDataUrl(file, maxDim = 1024, quality = 0.85) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the image file."));
    reader.onload = () => {
      const raw = reader.result;
      const img = new Image();
      img.onerror = () => resolve(raw);
      img.onload = () => {
        try {
          const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
          const w = Math.round(img.width * scale);
          const h = Math.round(img.height * scale);
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, w, h);
          const type = file.type === "image/png" ? "image/png" : "image/jpeg";
          resolve(canvas.toDataURL(type, quality));
        } catch {
          resolve(raw);
        }
      };
      img.src = raw;
    };
    reader.readAsDataURL(file);
  });
}
