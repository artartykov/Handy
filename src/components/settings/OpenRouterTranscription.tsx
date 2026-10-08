import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { Input } from "../ui/Input";
import { useSettings } from "../../hooks/useSettings";

// Fork-only: cloud transcription through OpenRouter. English text lives in the
// t() defaults so the fork doesn't touch every locale file.
const DEFAULT_MODEL = "openai/gpt-transcribe";

export const OpenRouterTranscription: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const enabled = getSetting("openrouter_stt_enabled") ?? false;
  const model = getSetting("openrouter_stt_model") ?? DEFAULT_MODEL;
  const [localModel, setLocalModel] = useState(model);

  useEffect(() => {
    setLocalModel(model);
  }, [model]);

  return (
    <SettingsGroup
      title={t("settings.models.openrouterStt.title", "Cloud transcription")}
    >
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("openrouter_stt_enabled", value)}
        isUpdating={isUpdating("openrouter_stt_enabled")}
        label={t(
          "settings.models.openrouterStt.enabled.label",
          "Transcribe with OpenRouter",
        )}
        description={t(
          "settings.models.openrouterStt.enabled.description",
          "Send recordings to OpenRouter instead of using a local model. Uses the OpenRouter API key from Post Processing settings.",
        )}
        descriptionMode="inline"
        grouped
      />
      <SettingContainer
        title={t("settings.models.openrouterStt.model.label", "Model")}
        description={t(
          "settings.models.openrouterStt.model.description",
          "OpenRouter speech-to-text model id.",
        )}
        descriptionMode="tooltip"
        grouped
        disabled={!enabled}
      >
        <Input
          type="text"
          value={localModel}
          onChange={(event) => setLocalModel(event.target.value)}
          onBlur={() => {
            const next = localModel.trim() || DEFAULT_MODEL;
            setLocalModel(next);
            if (next !== model) updateSetting("openrouter_stt_model", next);
          }}
          placeholder={DEFAULT_MODEL}
          variant="compact"
          disabled={!enabled || isUpdating("openrouter_stt_model")}
          className="min-w-[260px]"
        />
      </SettingContainer>
    </SettingsGroup>
  );
});

OpenRouterTranscription.displayName = "OpenRouterTranscription";
